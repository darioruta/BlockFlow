import torch


class Dao():
    def __init__(self):
        #self.repo_path = "/home/druta/models_repo"
        self.traced_path = "/home/druta/models_repo2/traced_repository"
        self.scripted_path = "/home/druta/models_repo2/scripted_repository"
        return
    
    def get_block(self, block_name, traced = True):
        # leggi modello da repository e ritornalo

        if traced:
            if block_name=="block1a":
                block1 = torch.jit.load(f'{self.traced_path}/traced_block1.pt')
                torch.jit.optimize_for_inference(block1)
                return block1
            elif block_name=="block1b":
                block1 = torch.jit.load(f'{self.traced_path}/traced_block1.pt')
                torch.jit.optimize_for_inference(block1)
                return block1
            elif block_name=="block1":
                block1 = torch.jit.load(f'{self.traced_path}/traced_block1.pt')
                torch.jit.optimize_for_inference(block1)
                return block1
            elif block_name=="block2a":
                block2 = torch.jit.load(f'{self.traced_path}/traced_block2.pt')
                torch.jit.optimize_for_inference(block2)
                return block2
            elif block_name=="block2b":
                block2 = torch.jit.load(f'{self.traced_path}/traced_block2.pt')
                torch.jit.optimize_for_inference(block2)
                return block2
            elif block_name=="block2":
                block2 = torch.jit.load(f'{self.traced_path}/traced_block2.pt')
                torch.jit.optimize_for_inference(block2)
                return block2
            elif block_name=="block3a":
                block3 = torch.jit.load(f'{self.traced_path}/traced_block3.pt')
                torch.jit.optimize_for_inference(block3)
                return block3
            elif block_name=="block3b":
                block3 = torch.jit.load(f'{self.traced_path}/traced_block3.pt')
                torch.jit.optimize_for_inference(block3)
                return block3
            elif block_name=="block3":
                block3 = torch.jit.load(f'{self.traced_path}/traced_block3.pt')
                torch.jit.optimize_for_inference(block3)
                return block3
            elif block_name=="block4a":
                block4 = torch.jit.load(f'{self.traced_path}/traced_block4.pt')
                torch.jit.optimize_for_inference(block4)
                return block4
            elif block_name=="block4b":
                block4 = torch.jit.load(f'{self.traced_path}/traced_block4.pt')
                torch.jit.optimize_for_inference(block4)
                return block4
            elif block_name=="block4":
                block4 = torch.jit.load(f'{self.traced_path}/traced_block4.pt')
                torch.jit.optimize_for_inference(block4)
                return block4
            elif block_name=="resnet50":
                resnet50 =  torch.jit.load(f"{self.traced_path}/traced_resnet.pt")
                torch.jit.optimize_for_inference(resnet50)
                return resnet50
            elif block_name=="resnet50a":
                resnet50 =  torch.jit.load(f"{self.traced_path}/traced_resnet.pt")
                torch.jit.optimize_for_inference(resnet50)
                return resnet50
            
            #block4+classifier
            elif block_name=="block4_class1":
                block4_class =  torch.jit.load(f"{self.traced_path}/traced_block4_class.pt")
                torch.jit.optimize_for_inference(block4_class)
                return block4_class
            elif block_name=="block4_class2":
                block4_class =  torch.jit.load(f"{self.traced_path}/traced_block4_class.pt")
                torch.jit.optimize_for_inference(block4_class)
                return block4_class
            elif block_name=="block4_class3":
                block4_class =  torch.jit.load(f"{self.traced_path}/traced_block4_class.pt")
                torch.jit.optimize_for_inference(block4_class)
                return block4_class
            
            #classification heads
            elif block_name=="class_head1" or block_name=="classifier1":
                classifier =  torch.jit.load(f"{self.traced_path}/traced_class_head.pt")
                torch.jit.optimize_for_inference(classifier)
                return classifier
            elif block_name=="class_head2" or block_name=="classifier2":
                classifier =  torch.jit.load(f"{self.traced_path}/traced_class_head.pt")
                torch.jit.optimize_for_inference(classifier)
                return classifier
            elif block_name=="class_head3" or block_name=="classifier3":
                classifier =  torch.jit.load(f"{self.traced_path}/traced_class_head.pt")
                torch.jit.optimize_for_inference(classifier)
                return classifier
            elif block_name=="class_head4" or block_name=="classifier4":
                classifier =  torch.jit.load(f"{self.traced_path}/traced_class_head.pt")
                torch.jit.optimize_for_inference(classifier)
                return classifier
            
            # detection heads
            elif block_name=="detection_head1" or block_name=="detector1":
                detector =  torch.jit.load(f"{self.traced_path}/traced_det_head.pt")
                torch.jit.optimize_for_inference(detector)
                return detector
            elif block_name=="detection_head2" or block_name=="detector2":
                detector =  torch.jit.load(f"{self.traced_path}/traced_det_head.pt")
                torch.jit.optimize_for_inference(detector)
                return detector
            elif block_name=="detection_head3" or block_name=="detector3":
                detector =  torch.jit.load(f"{self.traced_path}/traced_det_head.pt")
                torch.jit.optimize_for_inference(detector)
                return detector
            elif block_name=="detection_head4" or block_name=="detector4":
                detector =  torch.jit.load(f"{self.traced_path}/traced_det_head.pt")
                torch.jit.optimize_for_inference(detector)
                return detector

            else:
                print("uncorrect block name")
                return 
        else:
            #scripted branch
            if block_name=="block1":
                block1 = torch.jit.load(f'{self.scripted_path}/scripted_block1.pt')
                torch.jit.optimize_for_inference(block1)
                return block1
            elif block_name=="block1a":
                block1 = torch.jit.load(f'{self.scripted_path}/scripted_block1.pt')
                torch.jit.optimize_for_inference(block1)
                return block1
            elif block_name=="block1b":
                block1 = torch.jit.load(f'{self.scripted_path}/scripted_block1.pt')
                torch.jit.optimize_for_inference(block1)
                return block1
            elif block_name=="block2":
                block2 = torch.jit.load(f'{self.scripted_path}/scripted_block2.pt')
                torch.jit.optimize_for_inference(block2)
                return block2
            elif block_name=="block2a":
                block2 = torch.jit.load(f'{self.scripted_path}/scripted_block2.pt')
                torch.jit.optimize_for_inference(block2)
                return block2
            elif block_name=="block2b":
                block2 = torch.jit.load(f'{self.scripted_path}/scripted_block2.pt')
                torch.jit.optimize_for_inference(block2)
                return block2
            elif block_name=="block3":
                block3 = torch.jit.load(f'{self.scripted_path}/scripted_block3.pt')
                torch.jit.optimize_for_inference(block3)
                return block3
            elif block_name=="block3a":
                block3 = torch.jit.load(f'{self.scripted_path}/scripted_block3.pt')
                torch.jit.optimize_for_inference(block3)
                return block3
            elif block_name=="block3b":
                block3 = torch.jit.load(f'{self.scripted_path}/scripted_block3.pt')
                torch.jit.optimize_for_inference(block3)
                return block3
            elif block_name=="block4":
                block4 = torch.jit.load(f'{self.scripted_path}/scripted_block4.pt')
                torch.jit.optimize_for_inference(block4)
                return block4
            elif block_name=="block4a":
                block4 = torch.jit.load(f'{self.scripted_path}/scripted_block4.pt')
                torch.jit.optimize_for_inference(block4)
                return block4
            elif block_name=="block4b":
                block4 = torch.jit.load(f'{self.scripted_path}/scripted_block4.pt')
                torch.jit.optimize_for_inference(block4)
                return block4
            elif block_name=="resnet50":
                resnet50 = torch.jit.load(f"{self.scripted_path}/scripted_resnet.pt")
                torch.jit.optimize_for_inference(resnet50)
                return resnet50
            elif block_name=="resnet50a":
                resnet50 = torch.jit.load(f"{self.scripted_path}/scripted_resnet.pt")
                torch.jit.optimize_for_inference(resnet50)
                return resnet50
            

            #block4+classifier
            elif block_name=="block4_class1":
                block4_class =  torch.jit.load(f"{self.scripted_path}/scripted_block4_class.pt")
                torch.jit.optimize_for_inference(block4_class)
                return block4_class
            elif block_name=="block4_class2":
                block4_class =  torch.jit.load(f"{self.scripted_path}/scripted_block4_class.pt")
                torch.jit.optimize_for_inference(block4_class)
                return block4_class
            elif block_name=="block4_class3":
                block4_class =  torch.jit.load(f"{self.scripted_path}/scripted_block4_class.pt")
                torch.jit.optimize_for_inference(block4_class)
                return block4_class
            
            #classification heads
            elif block_name=="class_head1" or block_name=="classifier1":
                classifier =  torch.jit.load(f"{self.scripted_path}/scripted_class_head.pt")
                torch.jit.optimize_for_inference(classifier)
                return classifier
            elif block_name=="class_head2" or block_name=="classifier2":
                classifier =  torch.jit.load(f"{self.scripted_path}/scripted_class_head.pt")
                torch.jit.optimize_for_inference(classifier)
                return classifier
            elif block_name=="class_head3" or block_name=="classifier3":
                classifier =  torch.jit.load(f"{self.scripted_path}/scripted_class_head.pt")
                torch.jit.optimize_for_inference(classifier)
                return classifier
            elif block_name=="class_head4" or block_name=="classifier4":
                classifier =  torch.jit.load(f"{self.scripted_path}/scripted_class_head.pt")
                torch.jit.optimize_for_inference(classifier)
                return classifier
            
            # detection heads
            elif block_name=="detection_head1" or block_name=="detector1":
                detector =  torch.jit.load(f"{self.scripted_path}/scripted_det_head.pt")
                torch.jit.optimize_for_inference(detector)
                return detector
            elif block_name=="detection_head2" or block_name=="detector2":
                detector =  torch.jit.load(f"{self.scripted_path}/scripted_det_head.pt")
                torch.jit.optimize_for_inference(detector)
                return detector
            elif block_name=="detection_head3" or block_name=="detector3":
                detector =  torch.jit.load(f"{self.scripted_path}/scripted_det_head.pt")
                torch.jit.optimize_for_inference(detector)
                return detector
            elif block_name=="detection_head4" or block_name=="detector4":
                detector =  torch.jit.load(f"{self.scripted_path}/scripted_det_head.pt")
                torch.jit.optimize_for_inference(detector)
                return detector


            else:
                print("uncorrect block name")
                return 
        
    
    def save_new_block(self):
        return

