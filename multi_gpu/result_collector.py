import zmq
from tracker import MemoryTracker
import pickle
import time 
import torch
import json
import queue
import threading
import numpy as np

class Collector():

    def __init__(self, name, port, with_tcp, device, zero_copy, path_to_store):
        self.name = name
        self.port = port
        self.with_tcp = with_tcp
        self.device = device
        self.zero_copy = zero_copy
        self.path_to_store = path_to_store
        self.tracker = MemoryTracker(self.name)

        self.context = zmq.Context()
        self.incoming_socket = self.context.socket(zmq.PULL)

        if self.with_tcp:
            self.incoming_protocol =  "tcp"
            self.address = f"tcp://*:{self.port}"
            self.incoming_socket.bind(self.address)
            
        else:
            self.incoming_protocol =  "ipc"
            self.address = f"ipc:///tmp/{self.name}"
            self.incoming_socket.bind(self.address)
            
        print(f"Collector available at: {self.address}")
        self.client_connections = [] # list of dict: {"name": client1, "port": 5555, "socket": s}

        self.control_channel = self.context.socket(zmq.DEALER)
        self.control_channel.setsockopt(zmq.IDENTITY, self.serialize(self.name))
        #if self.with_tcp:
        self.control_address = "tcp://localhost:5600"
        #else:
        #    self.control_address = "ipc:///tmp/controller"
        self.control_channel.connect(self.control_address)

        self.data_queue = queue.Queue()
        self.control_queue = queue.SimpleQueue()

        self.delays = []
        self.times = []
        self.queue_sizes = []

        self.receiver_thread_on = True
        self.inference_thread_on = True
        self.queue_mon_thread_on = True
        self.control_thread_on = True

        self.lock = threading.Lock()


    def run(self):

        for i in range(3):
            torch.zeros(1,224,224, device=self.device)

        self.receiver = threading.Thread(target=self.receiver_thread, daemon=True)
        self.receiver.start()

        self.queue_mon = threading.Thread(target=self.queue_monitor, daemon=True)
        self.queue_mon.start()

        self.contr_thread = threading.Thread(target=self.control_thread, daemon=True)
        self.contr_thread.start()

        self.tracker.start()

        if self.zero_copy:
            while self.inference_thread_on: 
                
                arrival_time, sender, next_hops, data = self.data_queue.get()
                self.reconstruct_and_forward(arrival_time, sender, next_hops, data)
            
                
        else:
            while self.inference_thread_on: 
                
                arrival_time, sender, next_hops, data = self.data_queue.get()
                self.forward(arrival_time, sender, next_hops, data)
    

                
    

    def receiver_thread(self): 
        print(f"{self.name}: receiver thread started...")
        self.poller = zmq.Poller()
        self.poller.register(self.control_channel, zmq.POLLIN)
        self.poller.register(self.incoming_socket, zmq.POLLIN)
        while self.receiver_thread_on:
            events = dict(self.poller.poll(timeout=2))
            if self.incoming_socket in events and events[self.incoming_socket] == zmq.POLLIN:
                sender, next_hops, data = self.incoming_socket.recv_multipart()
                now = time.time()
                self.data_queue.put((now, sender, next_hops, data))
            if self.control_channel in events and events[self.control_channel] == zmq.POLLIN:
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
            return
        
        elif command == "info_time":
            out =  self.get_flying_stats()
            self.control_channel.send(self.serialize(out))
            return
        else:
            print(f"[{self.name}]: unrecognized control message!")
            return
    
    def get_flying_stats(self):
        return {}


    def reconstruct_and_forward(self, arrival_time, sender, next_hops, data):

        start = time.time()
        #deseralize
        des_data = self.deserialize(data)
        next_hops = self.deserialize(next_hops)
        check1 = time.time()

        #reconstruct
        handle_bytes = des_data["handle_bytes"]
        storage = torch.UntypedStorage._new_shared_cuda(*handle_bytes)
        tensor = torch.empty((0,), dtype=getattr(torch, des_data["dtype"]), device="cuda")
        tensor.set_(storage, 0, des_data["shape"], des_data["stride"])
        ready = time.time()


        #move to the cpu
        tensor = tensor.cpu()
        check2 = time.time()
        
        #serilize 
        serialized_tensor = self.serialize(tensor)
        check3 = time.time()

        #forward
        forwarding_socket = self.get_forwarding_socket(next_hops[0])
        forwarding_socket.send(serialized_tensor)
        end = time.time()

        del tensor
        

        total_processing =  round((end-start)*1000, 3)
        deserialization = round((check1-start)*1000,3)
        to_cpu = round((check2-ready)*1000, 3)
        reconstruction = round((ready-check1)*1000, 3)
        serialization = round((check3-check2)*1000,3)
        forwarding_time =  round((end-check3)*1000, 3)

        self.delays.append({"total_processing": total_processing,
                    "deserialization": deserialization,
                    "move_to_device": 0,

                    "reconstruction" : reconstruction,
                    "share_memory": 0,
                    "create_metadata": 0,

                    "processing_time": 0,
                    "serialization": serialization, 
                    "move_to_cpu": to_cpu,
                    "forwarding_time": forwarding_time
                    })
        waiting_time = round((start-arrival_time)*1000, 3)
        self.times.append({"arrival_time": arrival_time, "waiting": waiting_time, "processing": total_processing})

        return
    
    def forward(self, arrival_time, sender, next_hops, data):

        start = time.time()

        next_hops = self.deserialize(next_hops)
        check1 = time.time()

        client_info = next_hops[0]
        forwarding_socket = self.get_forwarding_socket(client_info)
        forwarding_socket.send(data)
        end = time.time()

        total_processing = round((end-start)*1000, 3)
        deserialization = round((check1- start)*1000, 3)
        forwarding = round((end-check1)*1000,3)


        self.delays.append({"total_processing": total_processing,
                    "deserialization": deserialization,
                    "move_to_device": 0,

                    "reconstruction" : 0,
                    "share_memory": 0,
                    "create_metadata": 0,

                    "processing_time": 0,
                    "serialization": 0, 
                    "move_to_cpu": 0,
                    "forwarding_time": forwarding
                    })
        
        waiting_time = round((start-arrival_time)*1000, 3)
        self.times.append({"arrival_time": arrival_time, "waiting": waiting_time, "processing": total_processing})
        return
    
    def store_kpi(self, sender):
        print(f"{self.name} -- CONTROL MESSAGE RECEIVED from {self.deserialize(sender)}:")
        skip_factor = 0

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
                        "reconstruction" : reconstruction_times,
                        "share_memory": share_memory_times,
                        "create_metadata": create_metadata_times,
                        "inference": inference_times,
                        "serialization": serialization_times,
                        "to_cpu": to_cpu,
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
        print(f"Block data stored at: {path}")
        with open(path, "w") as file:
            json.dump(stats, file, indent=3)
        file.close()

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
            test_duration = round(last_exit_time - first_arrival_time, 3)

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
        print(f"Block data stored at: {path}")
        with open(path, "w") as file:
            json.dump(res, file, indent=3)
        file.close()

        return 

    
    def get_forwarding_socket(self, client_info) -> zmq.Socket:
        for c  in self.client_connections:
            if c["name"] == client_info[0]:
                return c["socket"]
            
        print(f"Aggiungo una connessione TCP al client: {client_info}")
        new_socket = self.context.socket(zmq.PUSH)
        new_socket.connect(f"tcp://localhost:{client_info[1]}")
        self.client_connections.append({"name": client_info[0], "port": client_info[1], "socket": new_socket})
        return new_socket
    
    def serialize(self, tensor):
        return pickle.dumps(tensor, protocol=5)

    def deserialize(self, bytes_data):
        return pickle.loads(bytes_data)
    