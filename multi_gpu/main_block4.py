from block_dynamic import Block_dynamic_connections
import json
import os

if __name__ == "__main__": 
    with open("/home/druta/workspace/test/test_final_pipeline/base_config/block_port_translator.json", "r") as f:
        translator = json.load(f)
    f.close()

    with open("/home/druta/workspace/test/test_multi_GPU/base_config/block_device_translator_D.json", "r") as f:
        block_device_translator = json.load(f)
    f.close()

    BASEDIR = "/home/druta/workspace/test/test_multi_GPU/configD/"
    os.makedirs(os.path.dirname(BASEDIR), exist_ok=True)

    block_name = "block4"
    block_port = translator[block_name]
    with_tcp = True
    device = block_device_translator[block_name]
    zero_copy = True
    if zero_copy:
        path_to_store = BASEDIR+"zero_copy/"
        os.makedirs(os.path.dirname(path_to_store), exist_ok=True)
    else:
        path_to_store = BASEDIR

    b = Block_dynamic_connections(block_name, block_port, with_tcp, device, zero_copy, path_to_store, block_device_translator)
    b.run()
