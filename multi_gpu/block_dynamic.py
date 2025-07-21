from dao import Dao
from tracker import MemoryTracker
import torch
import zmq
import time
import pickle
import json
import threading 
import queue
import numpy as np
import sys
import os

class Block_dynamic_connections():

    def __init__(self, block_name, block_port, with_tcp, device, zero_copy, path_to_store, block_device_translator):

        self.connections = []
        self.block_name = block_name
        self.block_port = block_port
        self.with_tcp = with_tcp
        self.device = torch.device(device)
        self.cudaIPC = True
        self.dao = Dao()
        self.device = device
        torch.cuda.set_device(self.device)
        self.model = self.load_model()
        self.tracker = MemoryTracker(self.block_name, self.device)
        self.context = zmq.Context()
        self.incoming_socket = self.context.socket(zmq.PULL)
        self.incoming_socket.setsockopt(zmq.RCVHWM, 10000)
        if self.with_tcp:
            self.protocol = "tcp"
            self.incoming_socket.bind(f"tcp://*:{self.block_port}")
            self.incoming_socket.setsockopt(zmq.IDENTITY, self.serialize(self.block_name))
            self.block_address =  f"tcp://*:{self.block_port}"
            print(f"\n{block_name} correctly deployed -  tcp://localhost:{block_port} - device: {device}")
        else:
            self.protocol = "ipc"
            self.incoming_socket.bind(f"ipc:///tmp/{self.block_name}")
            self.incoming_socket.setsockopt(zmq.IDENTITY, self.serialize(self.block_name))
            self.block_address =  f"ipc:///tmp/{self.block_name}"
            print(f"\n{block_name} correctly deployed -  ipc:///tmp/{block_name} - device: {device}")

        self.path_to_store = path_to_store
        self.zero_copy = zero_copy

        self.control_channel = self.context.socket(zmq.DEALER)
        self.control_channel.setsockopt(zmq.IDENTITY, self.serialize(self.block_name))
        #if self.with_tcp:
        self.control_address = f"tcp://localhost:5600"
        self.control_channel.connect(self.control_address)
        #else:
            #self.control_address = f"ipc:///tmp/controller"
            #self.control_channel.connect(self.control_address)
        

        
        self.data_queue = queue.Queue()
        self.control_queue = queue.SimpleQueue()
        
        self.delays = [] # lista di dict dove vengono salvati i kpi di latenza per ogni inferenza
        self.times = [] # lista di dict dove vengono salvati i queuing times
        #self._flying_times = []  # lista di 100 elementi usata per monitoraggio istantaneo da parte del controller
        self.queue_sizes = [] # lista di monitoraggio size di coda messaggi
        
        self.receiver_thread_on = True
        self.inference_thread_on = True
        self.queue_mon_thread_on = True
        self.control_thread_on = True

        self.lock = threading.Lock()

        
        self.block_device_translator = block_device_translator
        


    def notify_readiness(self):
        self.control_channel.send(self.serialize("READY"))

    def run(self):
        #if self.zero_copy:
        #    self.run_with_zero_copy()
        #else:
        #    self.run_with_tensor_fwd()
        #torch.cuda.profiler()
        for i in range(3):
            torch.zeros(1,224,224, device=self.device)

        self.rec_thread = threading.Thread(target=self.receiver_thread, daemon=True)
        self.rec_thread.start()

        self.queue_mon_thread = threading.Thread(target=self.queue_monitor, daemon=True)
        self.queue_mon_thread.start()

        self.contr_thread = threading.Thread(target=self.control_thread, daemon=True)
        self.contr_thread.start()

        self.tracker.start()


        #self.notify_readiness()

        if self.zero_copy:
            i = 1
            with torch.no_grad():
                torch.cuda.synchronize()
                while self.inference_thread_on:
                    #try: 
                    #    control_msg = self.control_queue.get_nowait()
                    #    self.do_control(control_msg)
                    #except queue.Empty:
                    #    pass
                    
                    #try:
                    arrival_time, sender, next_hops, data = self.data_queue.get()
                    d = time.time() - arrival_time
                    #print(f"[{self.block_name}]: tensor {i} waited {round(d*1000, 2)} ms ")
                    self.do_inference_zero_copy(arrival_time, sender, next_hops, data)
                    #print(f"[{self.block_name}] zero copy inference {i} done")
                    i+=1
                    #except queue.Empty:
                    #    pass

                    
        else: 
            i = 1
            with torch.no_grad():
                torch.cuda.synchronize()
                while self.inference_thread_on:
                    #print(self.inference_thread_on)
                    #try: 
                    #    control_msg = self.control_queue.get_nowait()
                    #    self.do_control(control_msg)
                    #except queue.Empty:
                    #    pass

                    #try: 
                    arrival_time, sender, next_hops, data = self.data_queue.get()
                    d = time.time() - arrival_time
                    #print(f"[{self.block_name}]: tensor {i} waited {round(d*1000, 2)} ms ")
                    self.do_inference_tensor_fwd(arrival_time, sender, next_hops, data)
                    #print(f"[{self.block_name}] inference {i} done")
                    i+=1
                    #except queue.Empty:
                    #    pass
        
        print(f"[{self.block_name}] mi sto spegnendo")
        #self.store_kpi()
        #self.store_queuing_kpi()
        time.sleep(5)
        return
                    

    
    def do_inference_tensor_fwd(self, arrival_time, sender, next_hops, data): 
        #sender, next_hops, data = self.incoming_socket.recv_multipart()

        start = time.time()

        check1 = time.time()
        des_data = self.deserialize(data)
        next_hops = self.deserialize(next_hops)
    
        check2 = time.time()

        
        check2b = time.time()
        tensor = des_data.to(self.device)

        check3 = time.time()

        pred = self.model(tensor)

        check4 = time.time()

        pred = pred.cpu()
        check4b = time.time()
        serialized_pred = self.serialize(pred)
        check5 = time.time()
    
        forwarding_socket = self.get_forwarding_socket(next_hops[0])
        
        if len(next_hops)>1:
            forwarding_socket.send_multipart([self.serialize(self.block_name), self.serialize(next_hops[1:]), serialized_pred])
        else:
            forwarding_socket.send(serialized_pred)

        
        end = time.time()

        del pred
        del tensor

        latency =  round((end-start)*1000, 3)
        deserialization = round((check2-check1)*1000,3)
        moving_to_device = round((check3-check2b)*1000,3)
        processing_time = round((check4-check3)*1000,3)
        serialization = round((check5-check4b)*1000,3)
        move_to_cpu = round((check4b -check4)*1000, 3)
        forwarding_time =  round((end-check5)*1000, 3)

        self.delays.append({"total_processing": latency,
                    "deserialization": deserialization,
                    "move_to_device": moving_to_device,
                    "move_to_next_device": 0,

                    "reconstruction" : 0,
                    "share_memory": 0,
                    "create_metadata": 0,

                    "processing_time": processing_time,
                    "serialization": serialization,
                    "move_to_cpu": move_to_cpu,
                    "forwarding_time": forwarding_time})
        

        waiting_time = round((start-arrival_time)*1000, 3)
        #buffer_size = self.queue.qsize()
        self.times.append({"arrival_time": arrival_time, "waiting": waiting_time, "processing": latency})


        return
    
    
    def do_inference_zero_copy(self, arrival_time, sender, next_hops, data):
        start = time.time()
                
        des_data = self.deserialize(data)
        next_hops = self.deserialize(next_hops)
    
        check2 = time.time()

        handle_bytes = des_data["handle_bytes"]

        check3 = time.time()

        #reconstruct the tensor
        storage = torch.UntypedStorage._new_shared_cuda(*handle_bytes)
        tensor = torch.empty((0,), dtype=getattr(torch, des_data["dtype"]), device="cuda")
        tensor.set_(storage, 0, des_data["shape"], des_data["stride"])

        ready = time.time()

        pred = self.model(tensor)
        torch.cuda.synchronize()

        check4 = time.time()

        #move to the next-block device
        next_device =  self.get_next_device(next_hops[0])
        if next_device != self.device:
            pred = pred.to(next_device)
            torch.cuda.synchronize()
            check4a = time.time()
        else:
            check4a = check4

        storage = pred.untyped_storage()
        ipc_handle = storage._share_cuda_()

        check4b = time.time()
        

        metadata = {
            "shape": tuple(pred.shape),
            "dtype": str(pred.dtype).split('.')[-1], 
            "stride": tuple(pred.stride()),
            "handle_bytes" : ipc_handle
            }

        check5 = time.time()

        serialized_new_handle = self.serialize(metadata)

        check6 = time.time()

        forwarding_socket = self.get_forwarding_socket(next_hops[0])

        if len(next_hops)>1:
            forwarding_socket.send_multipart([self.serialize(self.block_name), self.serialize(next_hops[1:]), serialized_new_handle])
        else:
            #only the client/dispatcher remaining 
            forwarding_socket.send(serialized_new_handle)
        
        end = time.time()

        del pred
        del tensor
        torch.cuda.ipc_collect()

        total_processing =  round((end-start)*1000, 3)
        deserialization = round((check2-start)*1000,3)
        reconstruction = round((ready-check3)*1000, 3)
        processing_time = round((check4-ready)*1000,3)
        share_memory = round((check4b-check4)*1000, 3)
        create_metadata = round((check5-check4b)*1000, 3)
        serialization = round((check6-check5)*1000,3)
        forwarding_time =  round((end-check6)*1000, 3)
        move_to_next_device =  round((check4b-check4a)*1000,3)

        self.delays.append({"total_processing": total_processing,
                    "deserialization": deserialization,
                    "move_to_device": 0,

                    "reconstruction" : reconstruction,
                    "share_memory": share_memory,
                    "create_metadata": create_metadata,

                    "processing_time": processing_time,
                    "serialization": serialization, 
                    "move_to_cpu": 0,
                    "move_to_next_device": move_to_next_device,
                    "forwarding_time": forwarding_time
                    })
        
        waiting_time = round((start-arrival_time)*1000, 3)
        #buffer_size =  self.queue.qsize()
        self.times.append({"arrival_time": arrival_time, "waiting": waiting_time, "processing": total_processing})

        return


    def get_next_device(self,next_hop):
        return self.block_device_translator[next_hop[0]]
    
    def do_control(self, control_msg):
        print(f"[{self.block_name}] control message received: {self.deserialize(control_msg)}")

        control_msg = self.deserialize(control_msg)

        command = control_msg["command"]
        if command == "stop":
            print(f"[{self.block_name}]: Messaggio di stop ricevuto")
            with self.lock:
                self.queue_mon_thread_on = False
                self.receiver_thread_on = False
                self.inference_thread_on = False
            print(f"[{self.block_name}]: salvo i dati")
            self.store_kpi()
            #self.store_queuing_kpi()
            self.control_channel.send(self.serialize("Result stored and block stopped"))
            return
            
        elif command == "info_queue": #invia info su code per analisi tempi blocco
            print("sending queue state to the controller")
            self.control_channel.send(self.serialize(self.data_queue.qsize()))
            return
        elif command == "echo_request": #controlla se il blocco è attivo e invia indietro un ECHO reply
            print("ping received")
            self.control_channel.send(self.serialize("echo_reply"))
            print("reply sent")
            return

        elif command == "update":
            # chisura connessione e rimozione da lista self.connections per connessioni e a blocchi non piu attivi
            new_block_list = control_msg["new_block_list"]
            self.update_connections(new_block_list)
            self.control_channel.send(self.serialize("connections updated"))
            return
        
        elif command == "info_time":
            out =  self.get_flying_stats()
            self.control_channel.send(self.serialize(out))
            return
        else:
            print(f"[{self.block_name}]: unrecognized control message!")
            return
    

    def receiver_thread(self):
        print(f"{self.block_name}: receiver thread started...")
        self.poller = zmq.Poller()
        self.poller.register(self.control_channel, zmq.POLLIN)
        self.poller.register(self.incoming_socket, zmq.POLLIN)
        while self.receiver_thread_on:
            events = dict(self.poller.poll(timeout=2)) # 2ms
            if self.incoming_socket in events and events[self.incoming_socket] == zmq.POLLIN:
                sender, next_hops, data = self.incoming_socket.recv_multipart()
                now = time.time()
                self.data_queue.put((now, sender, next_hops, data))
            if self.control_channel in events  and events[self.control_channel] == zmq.POLLIN:
                print("nuovo messaggio di controllo ricevutooo")
                control_msg = self.control_channel.recv()
                self.control_queue.put(control_msg)
        print(f"[{self.block_name}]: receiver thread terminated...")


    def control_thread(self):
        print(f"[{self.block_name}]: control thread started...")
        while self.control_thread_on:
            msg = self.control_queue.get() # blocking quindi non consuma CPU
            self.do_control(msg)
        print(f"[{self.block_name}]: control thread terminated...")



    def queue_monitor(self):
        print(f"[{self.block_name}]: queue monitor thread started...")
        while self.queue_mon_thread_on:
            self.queue_sizes.append(self.data_queue.qsize())
            time.sleep(1)
        print(f"[{self.block_name}]: queue monitor thread terminated...")


    def load_model(self):
        model = self.dao.get_block(self.block_name)
        model = model.to(self.device).eval()
        return model
    
    def serialize(self, tensor):
        return pickle.dumps(tensor, protocol=5)

    def deserialize(self, bytes_data):
        return pickle.loads(bytes_data)

    
    def store_kpi(self):

        print(f"[{self.block_name}]: 'STORE' control message received!")
        skip_factor = 100

        stats = self.tracker.stop()
        time.sleep(1)

        stats["inferences_count"] = len(self.delays)

        total = [x["total_processing"] for x in self.delays]
        deserializations = [x["deserialization"] for x in self.delays]
        to_device = [x["move_to_device"] for x in self.delays]
        inference_times = [x["processing_time"] for x in self.delays]
        serialization_times = [x["serialization"] for x in self.delays]
        to_cpu = [x["move_to_cpu"] for x in self.delays]
        forwarding = [x["forwarding_time"] for x in self.delays]
        reconstruction_times =  [x["reconstruction"] for x in self.delays]
        share_memory_times = [x["share_memory"] for x in self.delays]
        create_metadata_times = [x["create_metadata"] for x in self.delays]
        move_to_next_device_times = [x["move_to_next_device"] for x in self.delays]

        
        num_inferences = len(self.times)

        if num_inferences>0:
            waiting_times = [x["waiting"] for x in self.times]
            processing_times = [x["processing"] for x in self.times]

            avg_waiting_time = np.mean(waiting_times[skip_factor:])
            avg_proc_time = np.mean(processing_times[skip_factor:])
            avg_time_spent_in_block = avg_waiting_time + avg_proc_time

            avg_buff_size = np.mean(self.queue_sizes)
            pending_requests = self.data_queue.qsize()
            first_arrival_time = self.times[skip_factor]["arrival_time"]
            last_exit_time = self.times[-1]["arrival_time"] + (self.times[-1]["waiting"])/1000 + (self.times[-1]["processing"])/1000
            test_duration = round(last_exit_time - first_arrival_time, 3)

            block_th = (num_inferences - skip_factor)/test_duration
        else:
            waiting_times = []
            processing_times = []

            avg_waiting_time = 0
            avg_proc_time = 0
            avg_time_spent_in_block = 0

            avg_buff_size = 0
            pending_requests = self.data_queue.qsize()
            first_arrival_time = 0
            last_exit_time = 0
            test_duration = 0

            block_th = 0

        #arrival rate 
        i = 0
        arrivals = []
        while i < len(self.times)-1:
            interarrival_time = self.times[i+1]["arrival_time"] - self.times[i]["arrival_time"]
            arrivals.append(interarrival_time * 1000)
            i+=1
        
        #avg_interarrival_time = np.mean(arrivals) # ms
        if len(arrivals) > 0:
            avg_interarrival_time = np.mean(arrivals) # ms
            arrival_rate = 1000/avg_interarrival_time #req/s
        else:
            avg_interarrival_time = 0
            arrival_rate = 0

        stats["times"] = {"total" : total,
                        "deserialization": deserializations, 
                        "move_to_device": to_device,
                        "move_to_next_device": move_to_next_device_times,
                        "reconstruction" : reconstruction_times,
                        "share_memory": share_memory_times,
                        "create_metadata": create_metadata_times,
                        "inference": inference_times,
                        "serialization": serialization_times,
                        "to_cpu": to_cpu,
                        "forwarding": forwarding}

        stats["queuing"] = {
            "test_duration_s": test_duration,
            "num_inf": num_inferences - skip_factor,
            "arrival_rate_req_s": round(arrival_rate,3),
            "avg_interarrival_time_ms": round(avg_interarrival_time,3),
            "avg_waiting_time_ms": round(avg_waiting_time, 3),
            "avg_proc_time_ms": round(avg_proc_time,3),
            "avg_time_spent" : avg_time_spent_in_block,
            "avg_buff_size": round(avg_buff_size,3),
            "pending": pending_requests,
            "block_th_req_s":  round(block_th, 3), 
            "waiting_times": waiting_times

        }

        path = self.path_to_store + f"{self.block_name}.json"
        print(f"Block data stored at: {path}")
        with open(path, "w") as file:
            json.dump(stats, file, indent=3)
        file.close()

        #time.sleep(3)
        #os._exit(0)

        return
    
    def store_queuing_kpi(self): 
        skip_factor = 0

        if self.tracker.is_running:
            _ = self.tracker.stop()

        time.sleep(1)

        num_inferences = len(self.times)

        if num_inferences>0:
            waiting_times = [x["waiting"] for x in self.times]
            processing_times = [x["processing"] for x in self.times]

            avg_waiting_time = np.mean(waiting_times[skip_factor:])
            avg_proc_time = np.mean(processing_times[skip_factor:])

            avg_buff_size = np.mean(self.queue_sizes)
            pending_requests = self.data_queue.qsize()
            first_arrival_time = self.times[skip_factor]["arrival_time"]
            last_exit_time = self.times[-1]["arrival_time"] + (self.times[-1]["waiting"])/1000 + (self.times[-1]["processing"])/1000
            test_duration = round(last_exit_time - first_arrival_time, 3)

            block_th = (num_inferences - skip_factor)/test_duration
        else:
            waiting_times = []
            processing_times = []

            avg_waiting_time = 0
            avg_proc_time = 0

            avg_buff_size = 0
            pending_requests = self.data_queue.qsize()
            first_arrival_time = 0
            last_exit_time = 0
            test_duration = 0

            block_th = 0

        #arrival rate 
        i = 0
        arrivals = []
        while i < len(self.times)-1:
            interarrival_time = self.times[i+1]["arrival_time"] - self.times[i]["arrival_time"]
            arrivals.append(interarrival_time * 1000)
            i+=1
        
        #avg_interarrival_time = np.mean(arrivals) # ms
        if len(arrivals) > 0:
            avg_interarrival_time = np.mean(arrivals) # ms
            arrival_rate = 1000/avg_interarrival_time #req/s
        else:
            avg_interarrival_time = 0
            arrival_rate = 0


        res = {
            "block_name": self.block_name,
            "test_duration_s": test_duration,
            "num_inf": num_inferences - skip_factor,

            "arrival_rate_req_s": round(arrival_rate,3),
            "avg_interarrival_time_ms": round(avg_interarrival_time,3),

            "avg_waiting_time_ms": round(avg_waiting_time, 3),
            "avg_proc_time_ms": round(avg_proc_time,3),
            "avg_buff_size": round(avg_buff_size,3),
            "pending": pending_requests,
            "block_th_req_s":  round(block_th, 3)
        }

        path = self.path_to_store + f"TH_{self.block_name}.json"
        print(f"{self.block_name} data stored at: {path}")
        with open(path, "w") as file:
            json.dump(res, file, indent=3)
        file.close()

        return 

    def get_flying_stats(self):
        return {}
 

    def get_forwarding_socket(self, next_hop) -> zmq.Socket: # next_hop is a tuple (hop_name, hop_port)

        for c in self.connections:
            if next_hop[0] == c["block_name"]:
                return c["socket"]
 
        print(f"[{self.block_name}]: Aggiungo nuova connessione al next hop: {next_hop}")
        new_socket = self.context.socket(zmq.PUSH)
        new_socket.setsockopt(zmq.SNDHWM, 10000)

        if self.protocol == "tcp":
            new_socket.connect(f"tcp://localhost:{next_hop[1]}")
            self.connections.append({"block_name": next_hop[0], "port": next_hop[1], "starting_time": time.time(), "socket": new_socket})
        else:
            new_socket.connect(f"ipc:///tmp/{next_hop[0]}")
            self.connections.append({"block_name": next_hop[0], "port": next_hop[1], "starting_time": time.time(), "socket": new_socket})

        return new_socket
    
    def update_connections(self, new_active_blocks_list):

        for idx in range(len(self.connections)-1, -1,-1):
            c = self.connections[idx]
            if c["block_name"] not in new_active_blocks_list:
                #to_remove.append(idx)
                c["socket"].close()
                removed = self.connections.pop(idx)
                print(f"Connection to {c["block_name"]} removed")

        del removed
        return
        
if __name__ == "__main__": 
    block_name = "resnet50"
    block_port = 5555
    with_tcp = False
    device = "cuda"
    zero_copy = False
    path_to_store = "/home/druta/workspace/"

    b = Block_dynamic_connections(block_name, block_port, with_tcp, device, zero_copy, path_to_store)
    b.run()
