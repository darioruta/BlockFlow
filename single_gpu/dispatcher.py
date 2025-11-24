import zmq
import pickle
import time
import torch
from tracker import MemoryTracker
import json 
import queue
import threading
import numpy as np

class Dispatcher():

    def __init__(self,name, port, with_tcp, zero_copy, device, path_to_store):
        
        #aggiungi meccanismo di traduzione
        self.name = name
        self.port = port
        self.tracker = MemoryTracker(self.name)
        self.context = zmq.Context()
        self.interface = self.context.socket(zmq.PULL)
        self.address = f"tcp://*:{self.port}"
        self.interface.bind(self.address)
        print(f"Dispatcher available at: {self.address}")
        if with_tcp:
            self.protocol = "tcp"
        else:
            self.protocol = "ipc"
            
        self.zero_copy = zero_copy
        self.device = device
        self.path_to_store = path_to_store

        self.control_channel = self.context.socket(zmq.DEALER)
        self.control_channel.setsockopt(zmq.IDENTITY, self.serialize(name))
        #if with_tcp:
        self.control_address = f"tcp://localhost:5600"
        self.control_channel.connect(self.control_address)
        #else:
        #    self.control_address = f"ipc:///tmp/controller"
        #    self.control_channel.connect(self.control_address)

        
        self.block_connections = [] # list of dict: {"name": block1, "port": 5555, "socket": s}

        self.data_queue = queue.Queue()
        self.control_queue = queue.SimpleQueue()
        

        self.delays = [] # lista di dict dove vengono salvati i kpi di latenza per ogni inferenza
        self.times = [] # lista di dict dove vengono salvati i queuing times
        self.queue_sizes = []  # lista di monitoraggio size di coda messaggi

        self.receiver_thread_on = True
        self.inference_thread_on = True
        self.queue_mon_thread_on = True
        self.control_thread_on = True

        self.lock = threading.Lock()

        self.task_path_translator = {} # key-value pairs -> "taskID": ["block1", "block2"] etc.. path
        

    def run(self):
        #if self.zero_copy:
        #    self.run_with_zero_copy()
        #else:
        #    self.run_with_tensor_fwd()
        for i in range(3):
            torch.zeros(1,224,224, device=self.device)

        self.rec_thread = threading.Thread(target=self.receiver_thread, daemon=True)
        self.rec_thread.start()

        self.queue_mon_thread = threading.Thread(target=self.queue_monitor, daemon=True)
        self.queue_mon_thread.start()

        self.contr_thread = threading.Thread(target=self.control_thread, daemon=True)
        self.contr_thread.start()

        self.tracker.start()

        torch.zeros(1,224,224, device=self.device)

        #notify readiness
        #self.notify_readiness()

        if self.zero_copy:
            i = 1
            while True:
                
                arrival_time, sender, path, tensor = self.data_queue.get()
                self.dispatch_handle(arrival_time, sender, path, tensor)
                print(f"[{self.name}] zero copy inference {i} dispatched")
                i+=1
        else:
            i = 1
            while True:
            
                arrival_time, sender, path, tensor = self.data_queue.get()
                self.dispatch_tensor(arrival_time, sender, path, tensor)
                print(f"[{self.name}] zero copy inference {i} dispatched")
                i+=1

                
                

    def dispatch_handle(self, arrival_time, sender, path, tensor):
        #self.tracker.start()
        start = time.time()

        tensor = self.deserialize(tensor)
        path = self.deserialize(path)
        
        check1 = time.time()
        tensor = tensor.to(self.device)
        torch.cuda.synchronize()

        check1a = time.time()
        storage = tensor.untyped_storage()
        ipc_handle = storage._share_cuda_()
        check2 = time.time()

        metadata = {
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype).split('.')[-1], 
            "stride": tuple(tensor.stride()),
            "handle_bytes" : ipc_handle
        }

        check3 = time.time()
        metadata = self.serialize(metadata)
        check4 = time.time()
        forwarding_socket = self.get_forwarding_socket(path[0])
        forwarding_socket.send_multipart([self.serialize(self.name), self.serialize(path[1:]), metadata])
        end = time.time()

        del tensor
        torch.cuda.ipc_collect()

        total_processing = round((end-start)*1000, 3)
        deserialization = round((check1 - start)*1000, 3)
        to_cuda = round((check1a - check1)*1000, 3)
        share_memory = round((check2 - check1a)*1000, 3)
        create_metadata = round((check3 - check2)*1000, 3)
        serialize = round((check4 - check3)*1000, 3)
        forwarding = round((end - check3)*1000, 3)

        self.delays.append({"total": total_processing,
                        "deserialization": deserialization,
                        "move_to_cuda": to_cuda, 
                        "share_memory": share_memory,
                        "create_metadata": create_metadata,
                        "serialization": serialize,
                        "forwarding": forwarding})
        
        waiting_time = round((start-arrival_time)*1000, 3)
        #buffer_size =  self.queue.qsize()
        self.times.append({"arrival_time": arrival_time, "waiting": waiting_time, "processing": total_processing})

        return 

    
    def dispatch_tensor(self, arrival_time, sender, path, tensor):
    
        start = time.time()

        path = self.deserialize(path)

        check1 = time.time()
        forwarding_socket =  self.get_forwarding_socket(path[0]) # (block_name, block_port)
        forwarding_socket.send_multipart([self.serialize(self.name), self.serialize(path[1:]), tensor])
        end = time.time()
        
        total_processing = round((end-start)*1000,3)
        deserialization = round((check1-start)*1000, 3)
        self.delays.append({"total": total_processing,
                            "deserialization": deserialization,
                            "move_to_cuda": 0, 
                            "share_memory": 0,
                            "create_metadata": 0,
                            "serialization": 0,
                            "forwarding": total_processing - deserialization})
        
        waiting_time = round((start-arrival_time)*1000, 3)
        #buffer_size =  self.queue.qsize()
        self.times.append({"arrival_time": arrival_time, "waiting": waiting_time, "processing": total_processing})

        return
        

            
                
    def do_control(self, control_msg):
        print(f"[{self.name}] control message received: {self.deserialize(control_msg)}")

        control_msg = self.deserialize(control_msg)

        command = control_msg["command"]
        if command == "stop":
            print(f"[{self.name}]: Messaggio di stop ricevuto")
            with self.lock:
                self.queue_mon_thread_on = False
                self.receiver_thread_on = False
                self.inference_thread_on = False
            print(f"[{self.name}]: salvo i dati")
            self.store_kpi()
            self.store_queuing_kpi()
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
            admitted_tasks = control_msg["admitted_task"]
            self.update_translator(admitted_tasks)
            self.control_channel.send(self.serialize("translator updated"))
            print(f"[{self.name}]: translator updated {json.dumps(self.task_path_translator, indent=3)}")
            return
        
        elif command == "info_time":
            out =  self.get_flying_stats()
            self.control_channel.send(self.serialize(out))
            return
        else:
            print(f"[{self.name}]: unrecognized control message!")
            return
    
    def update_translator(self, task_admitted):

        with self.lock:
            self.task_path_translator = task_admitted

        return
    
    def notify_readiness(self):
        self.control_channel.send(self.serialize("ready"))
        return
        

    def receiver_thread_old(self):
        print(f"[{self.name}]: receiver thread started...")
        self.poller = zmq.Poller()
        self.poller.register(self.control_channel, zmq.POLLIN)
        self.poller.register(self.interface, zmq.POLLIN)
        while self.receiver_thread_on:
            events = dict(self.poller.poll(timeout=2)) #2 ms
            if self.interface in events and events[self.interface] == zmq.POLLIN:
                sender, next_hops, data = self.interface.recv_multipart()
                now = time.time()
                self.data_queue.put((now, sender, next_hops, data))
            if self.control_channel in events  and events[self.control_channel] == zmq.POLLIN:
                control_msg = self.control_channel.recv()
                self.control_queue.put(control_msg)
        print(f"[{self.name}]: receiver thread terminated...")

    def receiver_thread(self):
        print(f"[{self.name}]: receiver thread started...")
        self.poller = zmq.Poller()
        self.poller.register(self.control_channel, zmq.POLLIN)
        self.poller.register(self.interface, zmq.POLLIN)
        while self.receiver_thread_on:
            events = dict(self.poller.poll(timeout=2)) #2 ms
            if self.interface in events and events[self.interface] == zmq.POLLIN:
                sender, data = self.interface.recv_multipart() 
                now = time.time()
                next_hops = self.task_path_translator[sender] #  NEW - TRANSLATION MECHANISM
                self.data_queue.put((now, sender, next_hops, data))
            if self.control_channel in events  and events[self.control_channel] == zmq.POLLIN:
                print(f"{self.name}: nuovo messaggio di controllo ricevutooo")
                control_msg = self.control_channel.recv()
                self.control_queue.put(control_msg)
        print(f"[{self.name}]: receiver thread terminated...")
        return

    def control_thread(self):
        print(f"[{self.name}]: control thread started...")
        while self.control_thread_on:
            msg = self.control_queue.get() # blocking quindi non consuma CPU
            self.do_control(msg)
        print(f"[{self.name}]: control thread terminated...")
        return

    def queue_monitor(self):
        print(f"[{self.name}]: queue monitor thread started...")
        while self.queue_mon_thread_on:
            self.queue_sizes.append(self.data_queue.qsize())
            time.sleep(1)
        print(f"[{self.name}]: queue monitor thread terminated...")
        return
                


    def get_forwarding_socket(self, node) -> zmq.Socket:
        
        for b in self.block_connections:
            if b["name"] == node[0]:
                return b["socket"]
        
        print(f"[DISPATCHER] Add a new connection to the first hop: {node}")
        socket = self.context.socket(zmq.PUSH)
        if self.protocol == "tcp":
            socket.connect(f"tcp://localhost:{node[1]}") # block_port
            self.block_connections.append({"name": node[0], "port": node[1],  "socket": socket})
        else:
            socket.connect(f"ipc:///tmp/{node[0]}") # block_name
            self.block_connections.append({"name": node[0], "port": node[1], "socket": socket})

        return socket



    

    def store_kpi(self, sender):
        print(f"{self.name} -- CONTROL MESSAGE RECEIVED from {self.deserialize(sender)}:")
        skip_factor = 0
        #print(f"\tConnections opened: {self.connections}\n")

        stats = self.tracker.stop()
        time.sleep(1)

        stats["inferences_count"] = len(self.delays)
        
        total = [x["total"] for x in self.delays]
        deserializations = [x["deserialization"] for x in self.delays]
        to_device = [x["move_to_cuda"] for x in self.delays]
        share_memory = [x["share_memory"] for x in self.delays]
        create_metadata = [x["create_metadata"] for x in self.delays]
        serialization = [x["serialization"] for x in self.delays]
        forwarding = [x["forwarding"] for x in self.delays]


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
                        "move_to_cuda": to_device,
                        "reconstruction" : 0,
                        "share_memory": share_memory,
                        "create_metadata": create_metadata,
                        "inference": 0,
                        "serialization": serialization,
                        "to_cpu": 0,
                        "forwarding": forwarding
                        }
        
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
        
        path = self.path_to_store + f"{self.name}.json"

        with open(path, "w") as f:
            json.dump(stats, f, indent=3)
        f.close()

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
            #buff_sizes = [x["buffer_size"] for x in self.times]
            
            avg_waiting_time = np.mean(waiting_times[skip_factor:])
            avg_proc_time = np.mean(processing_times[skip_factor:])
            #avg_buff_size = np.mean(buff_sizes[skip_factor:])
            avg_buff_size = np.mean(self.queue_sizes)
            pending_requests = self.data_queue.qsize()

            first_arrival_time = self.times[skip_factor]["arrival_time"]
            last_exit_time = self.times[-1]["arrival_time"] + (self.times[-1]["waiting"])/1000 + (self.times[-1]["processing"])/1000
            test_duration = last_exit_time - first_arrival_time

            block_th = (num_inferences - skip_factor) /test_duration
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
        
        if len(arrivals)>0:
            avg_interarrival_time = np.mean(arrivals) # ms
            arrival_rate = 1000/avg_interarrival_time #req/s
        else: 
            avg_interarrival_time = 0
            arrival_rate = 0

        res = {
            "block_name": self.name,
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

        path = self.path_to_store + f"TH_{self.name}.json"
        print(f"{self.name} data stored at: {path}")
        with open(path, "w") as file:
            json.dump(res, file, indent=3)
        file.close()

        return 

    def get_flying_stats(self):
        return {}
   
    def serialize(self, tensor):
        return pickle.dumps(tensor, protocol=5)

    def deserialize(self, bytes_data):
        return pickle.loads(bytes_data)
    

