import json
import numpy as np
from datetime import datetime


class StatsComputer():
    def __init__(self, load = False, input_path=None, input_data=None, output_path=None, test_configs=None):
        
        if load:
            print("loading data from filesystem")
            with open(input_path, "r") as f:
                self.data = json.load(f)
            f.close()
        else:
            print("data already available")
            self.data = input_data

        self.output_path = output_path
        self.test_configs = test_configs


    def compute_statistics(self):
        # Calculate total and peak memory by summing up the stats for each block process
        total_avg_ram = sum(stats['avg_ram_mb'] for stats in self.data if (stats["process"].startswith("b") or stats["process"].startswith("resnet")))
        peak_ram = sum(stats['peak_ram_mb'] for stats in self.data if (stats["process"].startswith("b") or stats["process"].startswith("resnet")))
        total_avg_vram = sum(stats['avg_vram_mb'] for stats in self.data if (stats["process"].startswith("b") or stats["process"].startswith("resnet")))
        peak_vram = sum(stats['peak_vram_mb'] for stats in self.data if (stats["process"].startswith("b") or stats["process"].startswith("resnet")))
        #total_gpu_cache = sum(stats["avg_gpu_cache_mb"] for stats in data if stats["process"].startswith("b"))
        
        cpu_usages = [stats["avg_cpu_usage"] for stats in self.data if (stats["process"].startswith("b") or stats["process"].startswith("resnet"))]
        cpu_clocks = [stats["avg_cpu_clock_mhz"] for stats in self.data if (stats["process"].startswith("b") or stats["process"].startswith("resnet"))]
        cpu_temps = [stats["avg_cpu_temp"] for stats in self.data if (stats["process"].startswith("b") or stats["process"].startswith("resnet"))]
        
        gpu_usages = [stats["avg_gpu_usage"] for stats in self.data if (stats["process"].startswith("b") or stats["process"].startswith("resnet"))]
        gpu_clocks = [stats["avg_gpu_clock_mhz"] for stats in self.data if (stats["process"].startswith("b") or stats["process"].startswith("resnet"))]
        gpu_temps = [stats["avg_gpu_temp"] for stats in self.data if (stats["process"].startswith("b") or stats["process"].startswith("resnet"))]
        avg_cpu_usage = np.mean(cpu_usages)
        avg_cpu_clock = np.mean(cpu_clocks)
        avg_cpu_temp = np.mean(cpu_temps)

        avg_gpu_usage = np.mean(gpu_usages)
        avg_gpu_clock = np.mean(gpu_clocks)
        avg_gpu_temp = np.mean(gpu_temps)

        tx_list = [stats["avg_pcie_tx_mb"] for stats in self.data if (stats["process"].startswith("b") or stats["process"].startswith("resnet"))]
        rx_list = [stats["avg_pcie_rx_mb"] for stats in self.data if (stats["process"].startswith("b") or stats["process"].startswith("resnet"))]
        avg_pcie_tx_rate = np.mean(tx_list)
        avg_pcie_rx_rate = np.mean(rx_list)

        # GPU CACHE
        gpu_cache_list = [stats["avg_gpu_cache_mb"] for stats in self.data if (stats["process"].startswith("b") or stats["process"].startswith("resnet"))]
        avg_gpu_cache_blocks = np.mean(gpu_cache_list)

        #times per each block
        skip_index = 5

        blocks_data = []
        clients_data = []
        dispatcher_data = []
        collector_data = []
        router_data = []
        client_list = []

        processes = [] 

        for proc in self.data: 
            processes.append(proc["process"])
            if proc["process"].startswith("client"): 
                c_name = proc["process"]
                print(f"Computing stats for...{c_name}")
                client_list.append(c_name)
                #avg_end_to_end = np.mean(proc["end_to_end"][skip_index:])
                #avg_waiting_time = np.mean(proc["waiting_time"][skip_index:])
                sending_rate = proc["sending_rate"]
                duration =  proc["duration"]
                tensors_received = proc["packets_received"]
                tensors_sent = proc["packets_sent"]
                avg_waiting_time_client = proc["avg_waiting_time"]
                empirical_th_measured  = proc["system_th"]
                avg_interarrival_time_client = proc["avg_interarrival_time"]
                c_data = { "client_name": c_name,
                        "sending_rate": sending_rate,
                        "duration": duration,
                        "tensors_received": tensors_received,
                        "tensors_sent": tensors_sent,
                        "measured_th": empirical_th_measured,
                        "avg_interarrival_time": avg_interarrival_time_client,
                        "avg_waiting_time": avg_waiting_time_client,
                        }
                clients_data.append(c_data)

            elif proc["process"].startswith("dispatcher"):
                d_name = proc["process"]
                print(f"Computing stats for...{d_name}")
                avg_ram_mb_d = proc["avg_ram_mb"]
                peak_ram_mb_d = proc["peak_ram_mb"]
                avg_vram_mb_d = proc["avg_vram_mb"]
                peak_vram_mb_d = proc["peak_vram_mb"]
                min_gpu_cache_d = proc["min_gpu_cache_mb"]
                max_gpu_cache_d = proc["max_gpu_cache_mb"]
                avg_gpu_cache_d= proc["avg_gpu_cache_mb"]

                avg_proc_cpu_usage = proc["avg_proc_cpu_usage"] 
                total_vol_ctx_s = proc["total_vol_ctx_switches"]
                total_invol_ctx_s = proc["total_invol_ctx_switches"]
                avg_vol_ctx_s_rate = proc["avg_vol_ctx_rate"]
                avg_invol_ctx_s_rate = proc["avg_invol_ctx_rate"]

                if len(proc["times"]["total"])>0: 
                    avg_time_spent_d_direct = np.mean(proc["times"]["total"][skip_index:]) #processing time
                    avg_deserialization_d = np.mean(proc["times"]["deserialization"][skip_index:])
                    avg_move_to_cuda_d = np.mean(proc["times"]["move_to_cuda"][skip_index:])
                    avg_share_memory_d = np.mean(proc["times"]["share_memory"][skip_index:])
                    avg_create_metadata_d = np.mean(proc["times"]["create_metadata"][skip_index:])
                    avg_serialization_d = np.mean(proc["times"]["serialization"][skip_index:])
                    avg_forwarding_d = np.mean(proc["times"]["forwarding"][skip_index:])

                    
                else:
                    avg_time_spent_d_direct = 0 #processing time
                    avg_deserialization_d = 0
                    avg_move_to_cuda_d = 0
                    avg_share_memory_d = 0
                    avg_create_metadata_d = 0
                    avg_serialization_d = 0
                    avg_forwarding_d = 0

                #QUEUING TIMES
                disp_time_alive_d = proc["queuing"]["test_duration_s"]
                num_inferences_done_d = proc["queuing"]["num_inf"]
                avg_arrival_rate_d = proc["queuing"]["arrival_rate_req_s"]
                avg_interarrival_time_d = proc["queuing"]["avg_interarrival_time_ms"]
                avg_waiting_time_d = proc["queuing"]["avg_waiting_time_ms"]
                avg_processing_time_d =  proc["queuing"]["avg_proc_time_ms"]
                avg_time_spent_d = proc["queuing"]["avg_time_spent"]
                avg_buff_size_d = proc["queuing"]["avg_buff_size"]
                pending_req_d = proc["queuing"]["pending"]
                disp_th_d = proc["queuing"]["block_th_req_s"]

                
                d_data = {"process": d_name,
                        "dispatcher_life" : disp_time_alive_d,
                        "num_inferences_done": num_inferences_done_d,
                        "throughput": disp_th_d,
                        "avg_arrival_rate": avg_arrival_rate_d,
                        "avg_interarrival_time" : avg_interarrival_time_d,
                        "avg_time_spent": round(avg_time_spent_d,3),
                        "avg_waiting_time": avg_waiting_time_d,
                        "avg_processing_time": avg_processing_time_d,
                        "avg_processing_time_direct" : round(avg_time_spent_d_direct, 3),
                        "avg_buffer_size": avg_buff_size_d,
                        "pending_requests": pending_req_d,
                        "avg_deserialization": round(avg_deserialization_d,3),
                        "avg_move_to_cuda": round(avg_move_to_cuda_d,3),
                        "avg_share_memory": round(avg_share_memory_d,3),
                        "avg_create_metadata": round(avg_create_metadata_d,3),
                        "avg_serialization": round(avg_serialization_d,3),
                        "avg_forwarding": round(avg_forwarding_d,3),

                        "avg_ram_mb" : round(avg_ram_mb_d,3),
                        "peak_ram_mb": round(peak_ram_mb_d,3),
                        "avg_vram_mb": round(avg_vram_mb_d,3),
                        "peak_vram_mb": round(peak_vram_mb_d,3),
                        "avg_gpu_cache": round(avg_gpu_cache_d,3),
                        "min_gpu_cache": round(min_gpu_cache_d,3),
                        "max_gpu_cache": round(max_gpu_cache_d,3),
                        
                        "avg_proc_cpu_usage" : round(avg_proc_cpu_usage,3),
                        "tot_vol_ctx_switch": round(total_vol_ctx_s,3),
                        "tol_invol_ctx_switch": round(total_invol_ctx_s,3),
                        "avg_vol_ctx_s_rate": round(avg_vol_ctx_s_rate,3),
                        "avg_invol_ctx_s_rate": round(avg_invol_ctx_s_rate,3),
                        }
                
                dispatcher_data.append(d_data)


            elif proc["process"].startswith("router"):
                r_name = proc["process"]
                print(f"Computing stats for...{r_name}")
                avg_forwarding_r = np.mean(proc["times"]["forwarding"][skip_index:])
                avg_reconstruction_r = np.mean(proc["times"]["reconstruction"][skip_index:])

                avg_ram_mb_r = proc["avg_ram_mb"]
                peak_ram_mb_r = proc["peak_ram_mb"]
                avg_vram_mb_r = proc["avg_vram_mb"]
                peak_vram_mb_r = proc["peak_vram_mb"]
                min_gpu_cache_r = proc["min_gpu_cache_mb"]
                max_gpu_cache_r = proc["max_gpu_cache_mb"]
                avg_gpu_cache_r= proc["avg_gpu_cache_mb"]

                avg_proc_cpu_usage = proc["avg_proc_cpu_usage"] 
                total_vol_ctx_s = proc["total_vol_ctx_switches"]
                total_invol_ctx_s = proc["total_invol_ctx_switches"]
                avg_vol_ctx_s_rate = proc["avg_vol_ctx_rate"]
                avg_invol_ctx_s_rate = proc["avg_invol_ctx_rate"]

                r_data = {
                    "process": r_name,
                    "avg_forwarding" : round(avg_forwarding_r,3),
                    "avg_reconstruction" : round(avg_reconstruction_r,3),

                    "avg_ram_mb" : round(avg_ram_mb_r,3),
                    "peak_ram_mb": round(peak_ram_mb_r,3),
                    "avg_vram_mb": round(avg_vram_mb_r,3),
                    "peak_vram_mb": round(peak_vram_mb_r,3),
                    "avg_gpu_cache": round(avg_gpu_cache_r,3),
                    "min_gpu_cache": round(min_gpu_cache_r,3),
                    "max_gpu_cache": round(max_gpu_cache_r,3),
                    "avg_proc_cpu_usage" : round(avg_proc_cpu_usage,3),
                    "tot_vol_ctx_switch": round(total_vol_ctx_s,3),
                    "tol_invol_ctx_switch": round(total_invol_ctx_s,3),
                    "avg_vol_ctx_s_rate": round(avg_vol_ctx_s_rate,3),
                    "avg_invol_ctx_s_rate": round(avg_invol_ctx_s_rate,3),
                }
                router_data.append(r_data)

            elif proc["process"].startswith("collector"):
                coll_name =  proc["process"]
                print(f"Computing stats for...{coll_name}")

                #if proc["inferences_count"]>0:
                
                print(coll_name)
                #print(json.dumps(proc, indent=3))
                avg_time_spent_c_direct =  np.mean(proc["times"]["total"][skip_index:])
                avg_deserialization_in_b  = np.mean(proc["times"]["deserialization"][skip_index:]) 
                avg_move_to_cuda_in_b =  np.mean(proc["times"]["move_to_device"][skip_index:])
                avg_reconstruction_in_b = np.mean(proc["times"]["reconstruction"][skip_index:])
                avg_share_memory_in_b = np.mean(proc["times"]["share_memory"][skip_index:])
                avg_create_metadata_in_b = np.mean(proc["times"]["create_metadata"][skip_index:]) 
                avg_inference_in_b = np.mean(proc["times"]["inference"][skip_index:]) 
                avg_serialization_in_b = np.mean(proc["times"]["serialization"][skip_index:])
                avg_to_cpu_in_b = np.mean(proc["times"]["to_cpu"][skip_index:])
                avg_forwarding_in_b = np.mean(proc["times"]["forwarding"][skip_index:])
                avg_ram_mb = proc["avg_ram_mb"]
                peak_ram_mb = proc["peak_ram_mb"]
                avg_vram_mb = proc["avg_vram_mb"]
                peak_vram_mb = proc["peak_vram_mb"]
                min_gpu_cache = proc["min_gpu_cache_mb"]
                max_gpu_cache = proc["max_gpu_cache_mb"]
                avg_gpu_cache = proc["avg_gpu_cache_mb"]
                avg_proc_cpu_usage = proc["avg_proc_cpu_usage"] 
                total_vol_ctx_s = proc["total_vol_ctx_switches"]
                total_invol_ctx_s = proc["total_invol_ctx_switches"]
                avg_vol_ctx_s_rate = proc["avg_vol_ctx_rate"]
                avg_invol_ctx_s_rate = proc["avg_invol_ctx_rate"]

                #QUEUING TIMES
                coll_time_alive = proc["queuing"]["test_duration_s"]
                num_inferences_done_c = proc["queuing"]["num_inf"]
                avg_arrival_rate_c = proc["queuing"]["arrival_rate_req_s"]
                avg_interarrival_time_c = proc["queuing"]["avg_interarrival_time_ms"]
                avg_waiting_time_c = proc["queuing"]["avg_waiting_time_ms"]
                avg_processing_time_c =  proc["queuing"]["avg_proc_time_ms"]
                avg_time_spent_c = proc["queuing"]["avg_time_spent"]
                avg_buff_size_c = proc["queuing"]["avg_buff_size"]
                pending_req_c = proc["queuing"]["pending"]
                coll_th_c = proc["queuing"]["block_th_req_s"]

            

                coll_data = {"process" :  coll_name, 
                        "collector_life": coll_time_alive,
                        "num_inferences_done" : num_inferences_done_c,
                        "throughput": coll_th_c,
                        "avg_arrival_rate": avg_arrival_rate_c,
                        "avg_interarrival_time" : avg_interarrival_time_c,
                        "avg_time_spent": avg_time_spent_c,
                        "avg_waiting_time": avg_waiting_time_c,
                        "avg_processing_time": avg_processing_time_c,
                        "avg_processing_time_direct" : round(avg_time_spent_c_direct, 3),
                        "avg_buffer_size": avg_buff_size_c,
                        "pending_requests": pending_req_c,
                        "avg_deserialization": round(avg_deserialization_in_b,3),
                        "avg_move_to_cuda": round(avg_move_to_cuda_in_b,3),
                        "avg_reconstruction" : round(avg_reconstruction_in_b,3),
                        "avg_share_memory" : round(avg_share_memory_in_b,3),
                        "avg_create_metadata" : round(avg_create_metadata_in_b,3),
                        "avg_inference" :  round(avg_inference_in_b,3),
                        "avg_serialization":  round(avg_serialization_in_b,3),
                        "avg_to_cpu" : round(avg_to_cpu_in_b,3),
                        "avg_forwarding" : round(avg_forwarding_in_b,3),
                        "avg_ram_mb" : round(avg_ram_mb,3),
                        "peak_ram_mb": round(peak_ram_mb,3),
                        "avg_vram_mb": round(avg_vram_mb,3),
                        "peak_vram_mb": round(peak_vram_mb,3),
                        "avg_gpu_cache": round(avg_gpu_cache,3),
                        "min_gpu_cache": round(min_gpu_cache,3),
                        "max_gpu_cache": round(max_gpu_cache,3),
                        "avg_proc_cpu_usage" : round(avg_proc_cpu_usage,3),
                        "tot_vol_ctx_switch": round(total_vol_ctx_s,3),
                        "tol_invol_ctx_switch": round(total_invol_ctx_s,3),
                        "avg_vol_ctx_s_rate": round(avg_vol_ctx_s_rate,3),
                        "avg_invol_ctx_s_rate": round(avg_invol_ctx_s_rate,3),

                        }
            
                collector_data.append(coll_data)


            else:
                b_name =  proc["process"]
                print(f"Computing stats for...{b_name}")
                avg_ram_mb = proc["avg_ram_mb"]
                peak_ram_mb = proc["peak_ram_mb"]
                avg_vram_mb = proc["avg_vram_mb"]
                peak_vram_mb = proc["peak_vram_mb"]
                min_gpu_cache = proc["min_gpu_cache_mb"]
                max_gpu_cache = proc["max_gpu_cache_mb"]
                avg_gpu_cache = proc["avg_gpu_cache_mb"]
                avg_proc_cpu_usage = proc["avg_proc_cpu_usage"] 
                total_vol_ctx_s = proc["total_vol_ctx_switches"]
                total_invol_ctx_s = proc["total_invol_ctx_switches"]
                avg_vol_ctx_s_rate = proc["avg_vol_ctx_rate"]
                avg_invol_ctx_s_rate = proc["avg_invol_ctx_rate"]

                #if proc["inferences_count"]>0:
                if len(proc["times"]["total"])>0:
                    #print(json.dumps(proc, indent=3))
                    avg_time_spent_b_direct =  np.mean(proc["times"]["total"][skip_index:])
                    avg_deserialization_in_b  = np.mean(proc["times"]["deserialization"][skip_index:]) 
                    avg_move_to_cuda_in_b =  np.mean(proc["times"]["move_to_device"][skip_index:])
                    avg_reconstruction_in_b = np.mean(proc["times"]["reconstruction"][skip_index:])
                    avg_share_memory_in_b = np.mean(proc["times"]["share_memory"][skip_index:])
                    avg_create_metadata_in_b = np.mean(proc["times"]["create_metadata"][skip_index:]) 
                    avg_inference_in_b = np.mean(proc["times"]["inference"][skip_index:]) 
                    avg_serialization_in_b = np.mean(proc["times"]["serialization"][skip_index:])
                    avg_to_cpu_in_b = np.mean(proc["times"]["to_cpu"][skip_index:])
                    avg_forwarding_in_b = np.mean(proc["times"]["forwarding"][skip_index:])
                    
                else:
                    avg_time_spent_b_direct =  0
                    avg_deserialization_in_b  = 0
                    avg_move_to_cuda_in_b =  0
                    avg_reconstruction_in_b = 0
                    avg_share_memory_in_b = 0
                    avg_create_metadata_in_b = 0 
                    avg_inference_in_b = 0
                    avg_serialization_in_b = 0
                    avg_to_cpu_in_b = 0
                    avg_forwarding_in_b = 0

                # BLOCK QUEUING TIMES
                block_time_alive_b = proc["queuing"]["test_duration_s"]
                num_inferences_done_b = proc["queuing"]["num_inf"]
                avg_arrival_rate_b = proc["queuing"]["arrival_rate_req_s"]
                avg_interarrival_time_b = proc["queuing"]["avg_interarrival_time_ms"]
                avg_waiting_time_b = proc["queuing"]["avg_waiting_time_ms"]
                avg_processing_time_b =  proc["queuing"]["avg_proc_time_ms"]
                avg_time_spent_b = proc["queuing"]["avg_time_spent"]
                avg_buff_size_b = proc["queuing"]["avg_buff_size"]
                pending_req_b = proc["queuing"]["pending"]
                block_th_b = proc["queuing"]["block_th_req_s"]


                b_data = {"process" :  b_name, 
                        "block_life" : block_time_alive_b,
                        "num_inferences_done": num_inferences_done_b,
                        "throughput" : block_th_b,
                        "avg_arrival_rate": avg_arrival_rate_b,
                        "avg_interarrival_time" : avg_interarrival_time_b,
                        "avg_time_spent": avg_time_spent_b,
                        "avg_waiting_time": avg_waiting_time_b,
                        "avg_processing_time": avg_processing_time_b,
                        "avg_processing_time_direct" : round(avg_time_spent_b_direct, 3),
                        "avg_buffer_size": avg_buff_size_b,
                        "pending_requests": pending_req_b,
                        "avg_deserialization": round(avg_deserialization_in_b,3),
                        "avg_move_to_cuda": round(avg_move_to_cuda_in_b,3),
                        "avg_reconstruction" : round(avg_reconstruction_in_b,3),
                        "avg_share_memory" : round(avg_share_memory_in_b,3),
                        "avg_create_metadata" : round(avg_create_metadata_in_b,3),
                        "avg_inference" :  round(avg_inference_in_b,3),
                        "avg_serialization":  round(avg_serialization_in_b,3),
                        "avg_to_cpu" : round(avg_to_cpu_in_b,3),
                        "avg_forwarding" : round(avg_forwarding_in_b,3),
                        "avg_ram_mb" : round(avg_ram_mb,3),
                        "peak_ram_mb": round(peak_ram_mb,3),
                        "avg_vram_mb": round(avg_vram_mb,3),
                        "peak_vram_mb": round(peak_vram_mb,3),
                        "avg_gpu_cache": round(avg_gpu_cache,3),
                        "min_gpu_cache": round(min_gpu_cache,3),
                        "max_gpu_cache": round(max_gpu_cache,3),
                        "avg_proc_cpu_usage" : round(avg_proc_cpu_usage,3),
                        "tot_vol_ctx_switch": round(total_vol_ctx_s,3),
                        "tol_invol_ctx_switch": round(total_invol_ctx_s,3),
                        "avg_vol_ctx_s_rate": round(avg_vol_ctx_s_rate,3),
                        "avg_invol_ctx_s_rate": round(avg_invol_ctx_s_rate,3)
                        }
                blocks_data.append(b_data)

        
        total_time_pipeline_blocks = sum(x["avg_time_spent"] for x in blocks_data)
        total_time_waiting = sum(x["avg_waiting_time"] for x in blocks_data)
        total_inference_blocks =  sum(x["avg_processing_time"] for x in blocks_data)
        total_deserialization =  sum(x["avg_deserialization"] for x in blocks_data)
        total_time_to_cuda =  sum(x["avg_move_to_cuda"] for x in blocks_data)
        total_computation = sum(x["avg_inference"] for x in blocks_data)
        total_serialization  = sum(x["avg_serialization"] for x in blocks_data)
        total_move_to_cpu  = sum(x["avg_to_cpu"] for x in blocks_data)
        total_forwarding = sum(x["avg_forwarding"] for x in blocks_data)

        l_serialization = [[x["avg_serialization"] for x in blocks_data]]
        l_deserialization = [x["avg_deserialization"] for x in blocks_data]
        l_forwarding = [x["avg_forwarding"] for x in blocks_data]
        avg_serialization = np.mean(l_serialization)
        avg_deserialization = np.mean(l_deserialization)
        avg_forwarding = np.mean(l_forwarding)


        print("formatting everything...")
        if "dispatcher" in processes:
            ram_dispatcher = d_data["avg_ram_mb"]
            vram_dispatcher = d_data["avg_vram_mb"]
            gpu_cache_dispatcher = d_data["avg_gpu_cache_mb"]
            dispatcher_th = d_data["throughput"]
        else:
            ram_dispatcher = 0
            vram_dispatcher = 0
            gpu_cache_dispatcher = 0
            dispatcher_th = 0

        if "router" in processes:
            print(json.dumps(r_data, indent=3))
            ram_router = r_data["avg_ram_mb"]
            vram_router = r_data["avg_vram_mb"]
            gpu_cache_router = r_data["avg_gpu_cache_mb"]
        else:
            ram_router = 0
            vram_router = 0
            gpu_cache_router = 0

        if "collector" in processes:
            ram_collector =  coll_data["avg_ram_mb"]
            vram_collector = coll_data["avg_vram_mb"]
            gpu_cache_collector = coll_data["avg_gpu_cache_mb"]
            collector_th = coll_data["throughput"]
        else:
            ram_collector = 0
            vram_collector = 0
            gpu_cache_collector = 0
            collector_th = 0
        
        
        tot_ram_system =  total_avg_ram + ram_dispatcher + ram_router + ram_collector
        tot_vram_system = total_avg_vram + vram_dispatcher + vram_router + vram_collector
        tot_gpu_cache_system = avg_gpu_cache_blocks + gpu_cache_dispatcher + gpu_cache_collector + gpu_cache_router

        blocks_th = [(x["process"], x["throughput"]) for x in blocks_data]
        pipeline_bottleneck =  min(blocks_th, key=lambda x: x[1])

        
        resources = {
            "total_avg_ram_mb_system": round(tot_ram_system,3),
            "total_avg_vram_mb_system" : round(tot_vram_system,3),
            "total_avg_gpu_cache_system": round(tot_gpu_cache_system,3),
            "total_avg_ram_mb_blocks": round(total_avg_ram,3),
            "peak_ram_mb_block": round(peak_ram,3),
            "total_avg_vram_mb_blocks": round(total_avg_vram,3),
            "peak_vram_mb_blocks": round(peak_vram, 3),
            "total_avg_gpu_cache_blocks": round(avg_gpu_cache_blocks,3),
            "avg_ram_dispatcher": ram_dispatcher,
            "avg_vram_dispatcher": vram_dispatcher,
            "avg_gpu_cache_dispatcher": gpu_cache_dispatcher,
            "avg_ram_collector" : ram_collector,
            "avg_vram_collector":  vram_collector,
            "avg_gpu_cache_collector": gpu_cache_collector,
            "avg_ram_router": ram_router,
            "avg_vram_router": vram_router,
            "avg_gpu_cache_collector": gpu_cache_router,
            "avg_cpu_usage": round(avg_cpu_usage,3),
            "avg_cpu_clock" : round(avg_cpu_clock,3),
            "avg_cpu_temp" : round(avg_cpu_temp),
            "avg_gpu_usage" : round(avg_gpu_usage,3),
            "avg_gpu_clock" : round(avg_gpu_clock),
            "avg_gpu_temp" : round(avg_gpu_temp,3),
            "avg_pcie_tx_rate_mb_s": round(avg_pcie_tx_rate,3),
            "avg_pcie_rx_rate_mb_s": round(avg_pcie_rx_rate,3)
        }

        sys_times = {
            "dispatcher_throughput": dispatcher_th,
            "collector_throughput": collector_th,
            "blocks_throughput": blocks_th,
            "pipeline_bottleneck": pipeline_bottleneck,
            "total_avg_time_pipeline": round(total_time_pipeline_blocks,3),
            "total_avg_time_waiting": round(total_time_waiting, 3),
            "total_avg_inference_blocks": round(total_inference_blocks,3),
            "total_avg_deserialization": round(total_deserialization,3),
            "total_avg_time_to_cuda" : round(total_time_to_cuda,3),
            "total_avg_computation" : round(total_computation,3),
            "total_avg_serialization" : round(total_serialization,3),
            "total_avg_time_to_cpu" : round(total_move_to_cpu,3),
            "total_avg_forwarding" : round(total_forwarding,3),
            "system_avg_serialization" : round(avg_serialization, 3),
            "system_avg_deserialization" : round(avg_deserialization, 3),
            "system_avg_forwarding" :  round(avg_forwarding,3)
        }

        test_stats = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "test_resume": self.test_configs,
            "sys_res": resources,
            "sys_times": sys_times,
            "blocks_data" : blocks_data, 
            "clients_data" : clients_data, 
            "dispatcher_data": dispatcher_data, 
            "router_data" : router_data,
            "collector_data": collector_data
        }


        #print(json.dumps(test_stats, indent=3))
        #path = self.output_path + f"{self.output_name}_stats.json"
        with open(self.output_path, "w") as f:
            json.dump(test_stats, f, indent=3)

        return

    def store_stats(self):
        return
    
    def compute_stats_dispatcher(self):
        return
    
    def compute_stats_block(self):
        return
    
    def compute_stats_collector(self):
        return
    
    def compute_stats_client(self):
        return

    
    