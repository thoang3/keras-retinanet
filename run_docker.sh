docker run -it --rm --runtime=nvidia \
    --shm-size 16G \
    --name RTNNKR \
    -u 0 \
    -v /home/tung/playground:/home/tung/playground \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -p 9095:6015 -p 5015:5015 \
    -e NVIDIA_VISIBLE_DEVICES=6 -e CUDA_DEVICE_ORDER=PCI_BUS_ID \
    nvcr.io/nvidia/tensorflow:19.09-py3 
