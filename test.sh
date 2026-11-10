# pre-trained from MVTec and ColonDB
ckt_path="weights/pretrained_mvtec_colondb.pth"
gpu_id=1

#CUDA_VISIBLE_DEVICES=$gpu_id python test.py --testing_model dataset --ckt_path $ckt_path --save_fig False --testing_data br35h   #ALL True before now is False
CUDA_VISIBLE_DEVICES=$gpu_id python test.py --testing_model dataset --ckt_path $ckt_path --save_fig False --testing_data brain_mri
CUDA_VISIBLE_DEVICES=$gpu_id python test.py --testing_model dataset --ckt_path $ckt_path --save_fig False --testing_data btad
CUDA_VISIBLE_DEVICES=$gpu_id python test.py --testing_model dataset --ckt_path $ckt_path --save_fig False --testing_data clinicdb
#CUDA_VISIBLE_DEVICES=$gpu_id python test.py --testing_model dataset --ckt_path $ckt_path --save_fig True --testing_data dagm
#CUDA_VISIBLE_DEVICES=$gpu_id python test.py --testing_model dataset --ckt_path $ckt_path --save_fig False --testing_data dtd
#CUDA_VISIBLE_DEVICES=$gpu_id python test.py --testing_model dataset --ckt_path $ckt_path --save_fig False --testing_data headct
#CUDA_VISIBLE_DEVICES=$gpu_id python test.py --testing_model dataset --ckt_path $ckt_path --save_fig False --testing_data isic
CUDA_VISIBLE_DEVICES=$gpu_id python test.py --testing_model dataset --ckt_path $ckt_path --save_fig False --testing_data mpdd
CUDA_VISIBLE_DEVICES=$gpu_id python test.py --testing_model dataset --ckt_path $ckt_path --save_fig False --testing_data sdd
#CUDA_VISIBLE_DEVICES=$gpu_id python test.py --testing_model dataset --ckt_path $ckt_path --save_fig False --testing_data tn3k
CUDA_VISIBLE_DEVICES=$gpu_id python test.py --testing_model dataset --ckt_path $ckt_path --save_fig False --testing_data visa

# pre-trained from Visa and Clinicdb
ckt_path="weights/pretrained_visa_clinicdb.pth"
gpu_id=1

CUDA_VISIBLE_DEVICES=$gpu_id python test.py --testing_model dataset --ckt_path $ckt_path --save_fig False --testing_data colondb
CUDA_VISIBLE_DEVICES=$gpu_id python test.py --testing_model dataset --ckt_path $ckt_path --save_fig False --testing_data mvtec



#nohup : nohup sh test.sh > "/home/spec/code/wjx/log/test.log" 2>&1 &



#dino on MVTec and ColonDB ��CUDA_VISIBLE_DEVICES=3 python test.py --testing_model dataset --ckt_path "weights/pretrained_mvtec_colondb.pth" --save_fig False --grounded_checkpoint "/mnt/md5/DL/code/wjx/AdaCLIP/method/GroundingDINO/grounding_weight/grounding_train_on_mvtec.pth" --testing_data visa 