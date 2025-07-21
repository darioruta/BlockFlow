from block_dynamic import Block_dynamic_connections
import json

if __name__ == "__main__": 
    with open("/home/druta/workspace/test/test_final_pipeline/base_config/block_port_translator.json", "r") as f:
        translator = json.load(f)
    f.close()
    BASEDIR = "/home/druta/workspace/test/test_final_pipeline/nvprof_test/"

    block_name = "block1"
    block_port = translator[block_name]
    with_tcp = True
    device = "cuda"
    zero_copy = True
    if zero_copy:
        path_to_store = BASEDIR+"zero_copy/"
    else:
        path_to_store = BASEDIR

    b = Block_dynamic_connections(block_name, block_port, with_tcp, device, zero_copy, path_to_store)
    b.run()
