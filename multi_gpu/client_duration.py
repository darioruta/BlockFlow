import zmq 
import time
import torch
import pickle
import json
import threading 
import random
import numpy as np
import sys

class ClientDuration():

    def __init__(self,name, port, with_tcp, path_to_store, zero_copy,path, num_inf, with_duration, duration, sending_rate, batch_size, with_dispatcher=False, with_collector=False, synchronous = False):

        self.name = name
        self.port = port

        self.context = zmq.Context()
        self.send_socket = self.context.socket(zmq.PUSH)
        self.recv_socket = self.context.socket(zmq.PULL)
        self.first_hop_connected = False
        self.zero_copy = zero_copy
        self.path_to_store = path_to_store
        self.with_tcp = with_tcp
        if self.with_tcp:
            self.test_protocol = "tcp"
        else:
            self.test_protocol = "ipc"

        self.path = path
        self.n_inf = num_inf
        self.batch_size = batch_size
        self.with_collector = with_collector
        self.with_dispatcher = with_dispatcher
        self.with_duration =  with_duration
        self.sending_rate = sending_rate
        self.duration = duration

        self.delays = []
        self.arrivals = []
        self.departures = []

        self.handle_size = []

        self.receiver_on = True
        self.lock = threading.Lock()

        self.syncronous = synchronous
        if not self.syncronous:
            self.receiver = threading.Thread(target=self.receiver_thread, daemon=True)


    def receiver_thread(self): 

        c = 0
        while self.receiver_on:
            res =  self.recv_socket.recv()
            if self.zero_copy:
                self.reconstruct_result(res) #MB
            now = time.time()
            self.arrivals.append(now)
            #print(f"message {c+1} arrived")
            c+=1
        print("receiver spento")
        time.sleep(3)
        return

    def reconstruct_result(self, data):
        des_data = self.deserialize(data)
        size_handle =  sys.getsizeof(des_data)
        print(f"handle size: {size_handle}")
        self.handle_size.append(size_handle)
        handle_bytes = des_data["handle_bytes"]
        #reconstruct the tensor
        storage = torch.UntypedStorage._new_shared_cuda(*handle_bytes)
        tensor = torch.empty((0,), dtype=getattr(torch, des_data["dtype"]), device="cuda:0")
        tensor.set_(storage, 0, des_data["shape"], des_data["stride"])
        torch.cuda.ipc_collect()
        del tensor

        return
    

    def generate_handles(self):
        i = 0
        start = time.monotonic()
        next_send_time = start
        #meccanismo di busy - wait
        while time.monotonic() - start < self.duration:
            now = time.monotonic()
            if now >= next_send_time:
                #msg = f"[{self.name}] "
                self.send_handle(i, self.batch_size, self.syncronous)
                i += 1
                if not self.syncronous:
                    # se non è sincrono attendi prima di inviare un nuovo messaggio
                    next_send_time += random.expovariate(self.sending_rate)

        return

    def generate_tensors(self):
        i = 0
        start = time.monotonic()
        next_send_time = start
        #meccanismo di busy - wait
        while time.monotonic() - start < self.duration:
            now = time.monotonic()
            if now >= next_send_time:
                self.send_tensor(i, self.batch_size, self.syncronous)
                i += 1
                if not self.syncronous:
                    # se non è sincrono attendi prima di inviare un nuovo messaggio
                    next_send_time += random.expovariate(self.sending_rate)

        return
    


        
    def run_always_tcp(self):
        #TUTTO TCP anche se protocol del sistema è IPC
        #zero_copy = False anche se zero_copy del sistema = True
        dispatcher_port = self.path[0][1]
        self.remaining_hops = self.path[1:]
        self.push_address = f"tcp://localhost:{dispatcher_port}"
        self.pull_address = f"tcp://*:{self.port}"
        self.send_socket.connect(self.push_address)
        self.recv_socket.bind(self.pull_address)
        print(f"client connected to: {self.push_address}")
        print(f"cleint available at: {self.pull_address}")
        time.sleep(1)

        print(f"{self.name} inference starting...")
        self.system_warmup()

        if not self.syncronous:
            self.receiver.start()
        time.sleep(1)

        self.generate_tensors()

        self.store_client_stats()
        self.close_connections()
        print(f"{self.name}: Results stored and connections closed")

        return
    
    def run_mixed_protocol(self):
        #self.forwarding_socket.connect(f"tcp://localhost:{dispatcher_port}")
        #self.incoming_socket.bind(.... according to the protocol)
        #1 caso: dispatcher = true , collector = false
        #2 caso: dispatcher = false, collector = true

        if self.with_dispatcher == True and self.with_collector == False:
            dispatcher_port = self.path[0][1]
            self.remaining_hops = self.path[1:]
            self.push_address = f"tcp://localhost:{dispatcher_port}"

            if self.test_protocol == "tcp":
                self.pull_address = f"tcp://*:{self.port}"
            else:
                self.pull_address = f"ipc:///tmp/{self.name}"

            self.send_socket.connect(self.push_address)
            self.recv_socket.bind(self.pull_address)
            print(f"client connected to: {self.push_address}")
            print(f"cleint available at: {self.pull_address}")
            #self.receiver = threading.Thread(target=self.receiver_thread, daemon=True)
            if not self.syncronous:
                self.receiver.start()
            time.sleep(1)

            self.generate_tensors()
            
            self.store_client_stats()
            self.close_connections()
            print(f"{self.name}: Results stored and connections closed")
            return

        
        if self.with_dispatcher == False and self.with_collector == True:
            first_hop_info = self.path[0]
            print(first_hop_info)
            self.remaining_hops = self.path[1:]
            if self.test_protocol == "tcp":
                self.push_address = f"tcp://localhost:{first_hop_info[1]}" #port
            else:
                self.push_address = f"ipc:///tmp/{first_hop_info[0]}" #name
            self.pull_address = f"tcp://*:{self.port}"
            self.send_socket.connect(self.push_address)
            self.recv_socket.bind(self.pull_address)
            print(f"client connected to: {self.push_address}")
            print(f"cleint available at: {self.pull_address}")
            #self.receiver = threading.Thread(target=self.receiver_thread, daemon=True)

            self.system_warmup()
            if not self.syncronous:
                self.receiver.start()
            time.sleep(1)
        
            if self.zero_copy:
                self.generate_handles()
            else:
                self.generate_tensors()
            
            self.store_client_stats()
            self.close_connections()
            print(f"{self.name}: Results stored and connections closed")
            return

            
    
    def run_standard(self):
        #protocollo dipende da quello del sistema
        #zero_copy dipende da quello del sistema
        first_hop_info = self.path[0]
        self.remaining_hops = self.path[1:]
        if self.test_protocol == "tcp":
            self.push_address = f"tcp://localhost:{first_hop_info[1]}" #port
            self.pull_address = f"tcp://*:{self.port}"
        else:
            self.push_address = f"ipc:///tmp/{first_hop_info[0]}" #name
            self.pull_address = f"ipc:///tmp/{self.name}"

        self.send_socket.connect(self.push_address)
        self.recv_socket.bind(self.pull_address)
        print(f"client connected to: {self.push_address}")
        print(f"cleint available at: {self.pull_address}")
        #self.receiver = threading.Thread(target=self.receiver_thread, daemon=True)

        self.system_warmup()

        if not self.syncronous:
                self.receiver.start()
        time.sleep(1)

        if self.zero_copy:
            self.generate_handles()
        else:
            self.generate_tensors()
            
        self.store_client_stats()
        self.close_connections()

        print(f"{self.name}: Results stored and connections closed")

        return
    
    def system_warmup(self):
        '''Warmup the pipeline connections and GPU. Respectful behavior - It waits the msg reception before sending the next one'''
        if self.zero_copy:
            for i in range(10):
                self.send_handle(i, self.batch_size, syncronous=True)
                torch.cuda.empty_cache()
        else:
            for i in range(10):
                self.send_tensor(i, self.batch_size, syncronous=True)
                torch.cuda.empty_cache()
    
    def run(self):
        
        if self.with_dispatcher:
            if self.with_collector:
                self.run_always_tcp()
            else:
                self.run_mixed_protocol()
        else:
            if self.with_collector:
                self.run_mixed_protocol()
            else:
                self.run_standard()
        return

    

    def close_connections(self):
        #self.recv_socket.unbind(self.address)
        self.send_socket.close()
        self.first_hop_connected = False

        return
    
    def send_handle(self,i, batch_size, syncronous):

        start = time.time()
        tensor = torch.rand(batch_size,3,224,224, device="cuda:0")
        torch.cuda.synchronize()
        check1 = time.time()

        storage = tensor.untyped_storage()
        ipc_handle = storage._share_cuda_()
        check2 = time.time()

        metadata = {
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype).split('.')[-1], 
            "stride": tuple(tensor.stride()),
            "handle_bytes" : ipc_handle
        }
        torch.cuda.synchronize()
        check3 = time.time()
        serialized = self.serialize(metadata)
        check4 = time.time()
        if self.with_dispatcher:
            self.send_socket.send_multipart([self.serialize(self.name), serialized])
        else:
            self.send_socket.send_multipart([self.serialize(self.name), self.serialize(self.remaining_hops), serialized])
        
        check5 = time.time()
        if syncronous:
            data = self.recv_socket.recv()
            end = time.time()
            self.reconstruct_result(data)
            self.arrivals.append(end)
            
        else:
            end = time.time()

        del tensor
        torch.cuda.ipc_collect()


        total =  round((end-start)*1000, 3)
        serialization = round((check4 - check3)*1000,3)
        share_memory = round((check2-check1)*1000,3)
        forwarding_time  = round((check5 - check4)*1000,3)
        receiving_time = round((end - check5)*1000, 3)
        if self.syncronous:
            print(f"{self.name}: inference {i+1} received in {total} ms - RTT: {forwarding_time + receiving_time} ms")

        self.delays.append({f"inference": i,
                    "total" : total,
                    "serialization": serialization,
                    "share_memory" : share_memory,
                    "forwarding_time": forwarding_time,
                    "receiving_time": receiving_time,
                    })
        self.departures.append(end)
        
        return
    
    def send_tensor(self, i, batch_size, syncronous):
        start = time.time()
        tensor = torch.rand(batch_size,3,224,224, device="cpu")
        check1 = time.time()
        serialized_tensor = self.serialize(tensor)
        check2 = time.time()
        if self.with_dispatcher:
            self.send_socket.send_multipart([self.serialize(self.name), serialized_tensor])
        else:
            self.send_socket.send_multipart([self.serialize(self.name), self.serialize(self.remaining_hops), serialized_tensor])
        
        check3 = time.time()
        if syncronous:
            data = self.recv_socket.recv()
            end = time.time()
            self.arrivals.append(end)
        else:
            end = time.time()
            
        total =  round((end-start)*1000, 3)
        serialization = round((check2 - check1)*1000,3)
        forwarding_time  = round((check3 - check2)*1000,3)
        receiving_time = round((end - check3)*1000, 3)
        if self.syncronous:
            print(f"{self.name}: inference {i+1} received in {total} ms - RTT: {(forwarding_time + receiving_time):3f} ms")
        
        self.delays.append({f"inference": i,
                    "total" : total,
                    "serialization": serialization,
                    "share_memory" : 0,
                    "forwarding_time": forwarding_time,
                    "receiving_time": receiving_time
                    })
        
        self.departures.append(end)

        return
    

    def store_client_stats(self):

        self.receiver_on = False
    
        # compute stats client
        generation_times_list = [x["total"] for x in self.delays] 
        serialization_list = [x["serialization"] for x in self.delays]
        forwarding_list = [x["forwarding_time"] for x in self.delays]
        #receiving_list = [x["receiving_time"] for x in self.delays]
        share_memory_list = [x["share_memory"] for x in self.delays]



        interarrival_times = []
        for i in range(len(self.arrivals)-1):
            t1 = self.arrivals[i]
            t2 = self.arrivals[i+1]
            d = (t2-t1)*1000
            interarrival_times.append(d)
        
        packets_received = len(self.arrivals)
        avg_interarrival_time = np.mean(interarrival_times) # real e2e delay per ogni task
        system_th = packets_received/self.duration
        avg_generation_time = np.mean(self.departures) 

        waiting_list = []
        #stima end2end delay
        for i in range(100,len(self.arrivals)):
            wait =  self.arrivals[i] - self.departures[i]
            wait = round((wait)*1000, 3)
            waiting_list.append(wait)

        avg_waiting_time =  np.mean(waiting_list)

        if self.zero_copy:
            avg_handle_size = np.mean(self.handle_size)
            std_handle_size = np.std(self.handle_size)
            max_handle_size = np.max(self.handle_size)
            min_handle_size = np.min(self.handle_size)
        else:
            avg_handle_size = 0
            std_handle_size = 0
            max_handle_size = 0
            min_handle_size = 0


        # append client stats
        client_time_stats = {"process": self.name, 
                             "sending_rate" : self.sending_rate,
                             "duration": self.duration,
                             "packets_received": packets_received,
                             "packets_sent": len(self.departures),
                             "avg_waiting_time": avg_waiting_time,
                             "system_th": system_th,
                            "avg_interarrival_time": avg_interarrival_time,
                            "serialization" : serialization_list, 
                            "share_memory": share_memory_list,
                            "forwarding" :  forwarding_list, 
                            "waiting_time": waiting_list, 
                            "arrival_list": self.arrivals, 
                            "avg_handle_size": float(avg_handle_size),
                            "std_handle_size": float(std_handle_size),
                            "max_handle_size": float(max_handle_size),
                            "min_handle_size": float(min_handle_size),
                            }
        

        with open(self.path_to_store + f"{self.name}.json", "w") as f:
            json.dump(client_time_stats, f, indent=3)
        f.close()

        return
    
    def serialize(self, tensor):
        return pickle.dumps(tensor, protocol=5)

    def deserialize(self, bytes_data):
        return pickle.loads(bytes_data)




    

    