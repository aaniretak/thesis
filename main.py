from utilities import modelInference
import traceback

task = input("Choose a model by typing the corresponding number: \n1.MobileNet V2\n2.ResNet50 V1\n3.Inception V3\n4.NASNet-A large\n5.EfficientNet V2 M")
#batch = int(input("Insert batch size"))
#if (batch == 1): batch = None

for batch in [2**i for i in range(6)]:

    # Parameters of each model
    # Format = [link,batch_input_shape,datasetPath,isBackgroundIncluded]
    MobileNet = ["https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4",[batch,224,224,3],'../Datasets/ImageNet',1]
    ResNet = ["https://tfhub.dev/tensorflow/resnet_50/classification/1",[batch,224,224,3],'../Datasets/ImageNet',0]
    Inception = ["https://tfhub.dev/google/imagenet/inception_v3/classification/5",[batch,299,299,3],'../Datasets/ImageNet',1]
    NASNet = ["https://tfhub.dev/google/imagenet/nasnet_large/classification/5",[batch,331,331,3],'../Datasets/ImageNet',1]
    EfficientNet = ["https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_m/classification/2",[batch,480,480,3],'../Datasets/ImageNet',0]

    model_dict = {'1':MobileNet,'2':ResNet,'3':Inception,'4':NASNet,'5':EfficientNet}

    try:
        inf_dict = modelInference(model_dict[task])
        inf_dict['batch'] = batch
        f = open("batchesInception.txt", "a")
        f.write(str(inf_dict)+'\n')
        f.close()
        print("For batchsize = {}:\n\tThoughput = {}samples/sec\n\tLatency(All) = {}sec\n------".format(batch,inf_dict['throughput'],sum(inf_dict['latency'])))
    except Exception:
        print(traceback.format_exc())
