import torch
import gc
import time
import os
from client_duration import  ClientDuration
from block_dynamic import Block_dynamic_connections
from dispatcher import Dispatcher
from result_collector import Collector
import torch.multiprocessing as mp
import zmq 
import json
import pickle
from stats_computer import StatsComputer


class TestPUSH():
    def __init__(self, configs, translator, numa):
        
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        time.sleep(3)
        torch.cuda.synchronize()

        self.configs = configs
        self.translator = translator
        self.test_name = configs["test_name"]
        self.with_tcp = configs["with_tcp"]
        self.family_test = configs["family_test"]
        self.blocks_active = []
        self.clients = []
        self.numa_node = numa
        

        script_dir = os.path.dirname(os.path.abspath(__file__)) 
        script_dir = script_dir + f"/results_numa{self.numa_node}/" + self.family_test
        self.save_dir = script_dir +"/"+ self.test_name
    
        self.raw_data_dir = self.save_dir + "/raw/" #...results/raw/
        self.aggregated_data_dir = self.save_dir + "/aggregated/" #...results/test_name/aggregated/
        self.final_dir = self.save_dir + "/final/" #.../test_name/final/

        os.makedirs(os.path.dirname(self.raw_data_dir), exist_ok=True)
        os.makedirs(os.path.dirname(self.aggregated_data_dir), exist_ok=True)
        os.makedirs(os.path.dirname(self.final_dir), exist_ok=True)

        if self.with_tcp:
            self.protocol = "tcp"
        else:
            self.protocol = "ipc"
        self.zero_copy = configs["zero_copy"]
        self.device = configs["device"]
        self.clients_configs = configs["clients"]

        self.with_dispatcher = configs["with_dispatcher"]
        if self.with_dispatcher:
            self.dispatcher_name = configs["dispatcher_name"]
            self.dispatcher_port = translator[self.dispatcher_name]
            self.deploy_dispatcher()
        else:
            self.dispatcher_name = None
            self.dispatcher_port = None

        self.with_collector = configs["with_collector"]
        if self.with_collector:
            self.collector_name = configs["collector_name"]
            self.collector_port = translator[self.collector_name]
            self.deploy_collector()
        else:
            self.collector_name = None
            self.collector_port = None
    
        self.sync = configs["synchronous"]

        for c_config in self.clients_configs:
            c_name = c_config["name"]
            c_port = c_config["port"]
            f_pass = c_config["path"]
            f_pass = [(b_name, self.translator[b_name]) for b_name in f_pass]
            batch = c_config["batch_size"]
            n_inf = c_config["n_inf"]
            duration = c_config["duration"]
            with_duration = c_config["with_duration"]
            sending_rate = c_config["sending_rate"]
            self.deploy_blocks(f_pass)
            self.prepare_client(c_name,c_port, f_pass, batch, self.sync, n_inf, with_duration, duration, sending_rate)



        print("Waiting for the actual startup of the blocks")
        time.sleep(5)


    def run_test(self):
        s = time.time()
        self.run_clients()
        e = time.time()
        print(f"client ended in {round((e-s),3)} s ")
        self.send_control_message_v2()
        time.sleep(5)
        self.terminate_blocks()
        time.sleep(5)
        self.aggregate_data()
        time.sleep(5)
        #self.compute_statistics()

        return
    
    def deploy_blocks(self, f_pass):
        for name, port in f_pass:
            if name not in [x["name"] for x in self.blocks_active]:
                print(f"Initializing {name}...")
                p = mp.Process(target=self.block_call, args=(name, port, self.with_tcp, self.device, self.zero_copy, self.raw_data_dir))
                p.start()
                self.blocks_active.append({"name": name,"port": port, "process": p, "active": True})

        return
    
    @staticmethod
    def block_call(name, port, with_tcp, device, zero_copy, path_to_store):
        block = Block_dynamic_connections(name, port, with_tcp, device, zero_copy, path_to_store)
        block.run()
        return
    
    def deploy_dispatcher(self):
        p = mp.Process(target=self.dispatcher_call, args=(self.dispatcher_name, self.dispatcher_port, self.with_tcp, self.zero_copy, self.device, self.raw_data_dir))
        p.start()
        self.blocks_active.append({"name": self.dispatcher_name, "port": self.dispatcher_port, "process": p, "active": True})
        return
    
    @staticmethod
    def dispatcher_call(name, port, with_tcp, zero_copy, device, path_to_store):
        disp = Dispatcher(name, port, with_tcp, zero_copy, device, path_to_store)
        disp.run()
        return
    
    def deploy_collector(self):
        p = mp.Process(target=self.collector_call, args=(self.collector_name, self.collector_port, self.with_tcp, self.device,  self.zero_copy,  self.raw_data_dir))
        p.start()
        self.blocks_active.append({"name": self.collector_name, "port": self.collector_port, "process": p, "active": True})
        
        return
    
    @staticmethod
    def collector_call(name, port, with_tcp, device, zero_copy, path_to_store):
        collector = Collector(name, port, with_tcp, device, zero_copy, path_to_store)
        collector.run()
        return
    
    def terminate_blocks(self):
        for i in range(len(self.blocks_active)):
            self.blocks_active[i]["process"].terminate()
            self.blocks_active[i]["active"] = False

        return
    
    def send_control_message(self):
        context = zmq.Context()
        for b in self.blocks_active:
            if b["name"].startswith("dispatcher")==False:
                sender = context.socket(zmq.PUSH)
                if self.with_tcp:
                    address = f"tcp://localhost:{b['port']}"
                    #print(address)
                else:
                    address = f"ipc:///tmp/{b['name']}"

                sender.connect(address)
                sender.send_multipart([self.serialize("controller"), self.serialize("no_hop"), self.serialize("control")])
                sender.close()
            else:
                # Il dispatcher riceve sempre su socket TCP perche funge da entrypoint per il mondo esterno
                sender = context.socket(zmq.PUSH)
                address = f"tcp://localhost:{b['port']}"
                sender.connect(address)
                sender.send_multipart([self.serialize("controller"), self.serialize("no_hop"), self.serialize("control")])
                sender.close()

        return
    

    def send_control_message_v2(self):
        context = zmq.Context()
        socket = context.socket(zmq.ROUTER)
        socket.setsockopt(zmq.IDENTITY, b"controller")
        #if self.with_tcp:
        socket.bind("tcp://*:5600")
        #else:
        #    socket.bind("ipc:///tmp/controller")

        time.sleep(3)
        
        print("invio messaggi di controllo")
        
        msg = {"command": "stop"}
        for b in self.blocks_active:
            socket.send_multipart([self.serialize(b["name"]), self.serialize(msg)])


        socket.close()

        return
    
    def prepare_client(self, name, port, f_pass, batch_size, sync, n_inf = None, with_duration =None, duration = None, sending_rate=None):
        if self.with_dispatcher:
            f_pass.insert(0, (self.dispatcher_name, self.dispatcher_port))
        
        if self.with_collector:
            f_pass.append((self.collector_name, self.collector_port))
        
        f_pass.append((name,port)) # client info at the end 

        print(f"{name} actual forward pass: {f_pass}")
        c = mp.Process(target=self.client_call, args=(name, port, self.with_tcp, self.raw_data_dir, self.zero_copy, f_pass, n_inf, with_duration, duration, sending_rate, batch_size, self.with_dispatcher, self.with_collector, sync))
        self.clients.append({"name": name, "port": port, "process": c, "active": False})
        return
    
    @staticmethod
    def client_call(name, port, with_tcp, path_to_store, zero_copy, path, n_inf, with_duration, duration, sending_rate, batch_size, with_dispatcher, with_collector, sync):
        #name, port, with_tcp, path_to_store, zero_copy,path, num_inf, with_duration, duration, sending_rate, batch_size, with_dispatcher=False, with_collector=False
        c = ClientDuration(name, port, with_tcp, path_to_store, zero_copy, path, n_inf, with_duration, duration, sending_rate, batch_size, with_dispatcher, with_collector, sync)
        c.run()
        return
    
    def run_clients(self):
        print("Starting client requests...")
        for c in self.clients:
            c["process"].start()
            c["active"] = True

        for c in self.clients:
            c["process"].join()
            c["active"] = False
        return
    
    def aggregate_data(self):
        self.aggregated_stats = []
        self.aggregated_th_buffer_stats = []
        for b in self.blocks_active:
            try:
                with open(self.raw_data_dir + f"/{b['name']}.json", "r") as f:
                    self.aggregated_stats.append(json.load(f))
                    f.close()
            except:
                print(f"Could not read stats for {b['name']}")
            
            try:
                with open(self.raw_data_dir + f"/TH_{b['name']}.json", "r") as f:
                    self.aggregated_th_buffer_stats.append(json.load(f))
                    f.close()
            except:
                print(f"Could not read TH stats for {b['name']}")


        for c in self.clients:
            try:
                with open(self.raw_data_dir + f"/{c['name']}.json", "r") as f:
                    self.aggregated_stats.append(json.load(f))
                    f.close()
            except:
                print(f"Could not read stats for {c['name']}")

        file_path = self.aggregated_data_dir + f"{self.test_name}.json"
        with open(file_path, "w") as f:
            json.dump(self.aggregated_stats, f, indent=3)
        f.close()

        file_path = self.aggregated_data_dir + f"TH_{self.test_name}.json"
        with open(file_path, "w") as f:
            json.dump(self.aggregated_th_buffer_stats, f, indent=3)
        f.close()

        print(f"Stats correctly aggregated and stored at {file_path}")
        time.sleep(3)
        return

    def serialize(self, data):
        return pickle.dumps(data, protocol=5)

    def deserialize(self, bytes_data):
        return pickle.loads(bytes_data)
    
    def compute_statistics(self):
        file_path = self.aggregated_data_dir + f"{self.test_name}.json"
        output_path = self.final_dir + f"{self.test_name}_stats.json"
        #computer = StatsComputer(file_path, output_path, self.configs)
        computer = StatsComputer(load=False, input_path=None, input_data=self.aggregated_stats, output_path=output_path, test_configs=self.configs)
        computer.compute_statistics()
        print(f"Final stats available at {file_path}")
        return
    
def doTest1(translator, numa):
    with open("/home/druta/workspace/test/test_final_pipeline/test_configs/test1_time_pipeline.json", "r") as f:
        configs = json.load(f)
    f.close()

    for t_conf in configs:
        print(f"\n\n #### STARTING NEW TEST #### \n\n {t_conf["test_name"]}")
        t = TestPUSH(t_conf, translator, numa)
        t.run_test()
        time.sleep(5)
    return


def doTest3(translator, numa):
    '''TEST 3: evaluate the system performance under varible batch sizes in order to compare tensorMQ and zeroMQ '''
    #with open("/home/druta/workspace/test/test_final_pipeline/test_configs/test3_scalability_batch.json", "r") as f:
    #    configs = json.load(f)
    #f.close()

    #for t_conf in configs:
    #    print(f"\n\n #### STARTING NEW TEST #### \n\n {t_conf["test_name"]}")
    #    t = TestPUSH(t_conf, translator, numa)
    #    t.run_test()
    #    time.sleep(5)


    with open("/home/druta/workspace/test/test_final_pipeline/test_configs/test3_scalability_batch_zero_copy.json", "r") as f:
        configs = json.load(f)
    f.close()

    for t_conf in configs:
        print(f"\n\n #### STARTING NEW TEST #### \n\n {t_conf["test_name"]}")
        t = TestPUSH(t_conf, translator, numa)
        t.run_test()
        time.sleep(5)

    return

def doTest4(translator, zero_copy, numa):
    '''Test 4: maximum throughput test. stats collected by varying the arrival rate to the pipeline'''
    if zero_copy:
        with open("/home/druta/workspace/test/test_final_pipeline/test_configs/test4_maximum_th_zero_copy.json", "r") as f:
            configs = json.load(f)
        f.close()

        for t_conf in configs:
            print(f"\n\n #### STARTING NEW TEST #### \n\n {t_conf["test_name"]}")
            t = TestPUSH(t_conf, translator, numa)
            t.run_test()
            time.sleep(5)
    else:
        with open("/home/druta/workspace/test/test_final_pipeline/test_configs/test4_maximum_th.json", "r") as f:
            configs = json.load(f)
        f.close()

        for t_conf in configs:
            print(f"\n\n #### STARTING NEW TEST #### \n\n {t_conf["test_name"]}")
            t = TestPUSH(t_conf, translator, numa)
            t.run_test()
            time.sleep(5)

    return

def doTest5(translator, zero_copy, numa):
    '''Test5: multiclient system scalability with clients adopting syncronous behavior'''
    if zero_copy:
        with open("/home/druta/workspace/test/test_final_pipeline/test_configs/test5_multiclient_scalability_zero_copy.json", "r") as f:
            configs = json.load(f)
        f.close()

        for t_conf in configs:
            print(f"\n\n #### STARTING NEW TEST #### \n\n {t_conf["test_name"]}")
            t = TestPUSH(t_conf, translator, numa)
            t.run_test()
            time.sleep(5)
    else:
        with open("/home/druta/workspace/test/test_final_pipeline/test_configs/test5_multiclient_scalability.json", "r") as f:
            configs = json.load(f)
        f.close()

        for t_conf in configs:
            print(f"\n\n #### STARTING NEW TEST #### \n\n {t_conf["test_name"]}")
            t = TestPUSH(t_conf, translator, numa)
            t.run_test()
            time.sleep(5)

    return

def main(path):
    with open(path, "r") as f:
        configs = json.load(f)
    f.close()

    for t_conf in configs:
        print(f"\n\n #### STARTING NEW TEST #### \n\n {t_conf["test_name"]}")
        t = TestPUSH(t_conf, translator)
        t.run_test()
        time.sleep(5)

    return

    

if __name__=="__main__":
    import sys
    mp.set_start_method("spawn")

    with open("/home/druta/workspace/test/test_final_pipeline/base_config/block_port_translator.json", "r") as f:
        translator = json.load(f)
    f.close()

    #main(path="/home/druta/workspace/test/test_final_pipeline/base_config/test_config.json")

    numa = sys.argv[1]
    print(f"Running numa node: {numa}")

    ###### TEST 1 ######
    doTest1(translator, numa)

    ###### TEST 3 ######
    doTest3(translator, numa)
    
    ###### TEST 4 ######
    doTest4(translator, zero_copy=False, numa=numa)
    
    ###### TEST 4 - ZERO COPY  ######
    doTest4(translator, zero_copy=True, numa=numa)
    
    ###### TEST 5 ######
    doTest5(translator, zero_copy=False, numa=numa)

    ###### TEST 5 ZERO COPY ######
    doTest5(translator, zero_copy=True, numa=numa)

    

    
