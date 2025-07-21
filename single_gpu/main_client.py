from client_duration import ClientDuration
import json
import zmq
import time
import pickle

def serialize(data):
        return pickle.dumps(data, protocol=5)

def send_control_message_v2(path):
        context = zmq.Context()
        socket = context.socket(zmq.ROUTER)
        socket.setsockopt(zmq.IDENTITY, b"controller")

        socket.bind("tcp://*:5600")
   
        time.sleep(3)
        
        print("invio messaggi di controllo")
        
        msg = {"command": "stop"}
        for b in path:
            if b[0] != "client1":
                socket.send_multipart([serialize(b), serialize(msg)])

        socket.close()

        return


if __name__ == "__main__":
    with open("/home/druta/workspace/test/test_final_pipeline/base_config/block_port_translator.json", "r") as f:
        translator = json.load(f)
    f.close()

    BASEDIR = "/home/druta/workspace/test/test_final_pipeline/nvprof_test2/"

    name = "client1"
    port = 5590
    with_tcp = True
    zero_copy = True
    forward_pass = ["block1", "block2","block3","block4"]
    n_it = 0
    with_duration= True
    duration = 60
    sending_rate = 0
    batch_size = 1
    with_dispatcher = False
    with_collector = False
    synchronous = True

    new_forward_pass = []
    for b in forward_pass: 
        temp = (b, translator[b])
        new_forward_pass.append(temp)
   
    new_forward_pass.append(("client1", 5590))

    if zero_copy:
        path_to_store = BASEDIR+"zero_copy/"
    else:
        path_to_store = BASEDIR

    client = ClientDuration(name, port,with_tcp, path_to_store,
                            zero_copy, new_forward_pass, n_it, with_duration,
                            duration, sending_rate,batch_size,
                            with_dispatcher, with_collector, synchronous)
    client.run()
    send_control_message_v2(forward_pass)