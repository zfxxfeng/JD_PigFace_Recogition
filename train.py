import tensorflow as tf
from input_Image_preprocessing import preprocess_for_train
import resnet_inference
import os
import time

files = tf.train.match_filenames_once("./shuffle_data_few_tfrecords/data.tfrecords-*")
filename_queue = tf.train.string_input_producer(files,shuffle=True)

reader = tf.TFRecordReader()
_,serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(
    serialized_example,
    features={
        'image_raw':tf.FixedLenFeature([],tf.string),
        'label':tf.FixedLenFeature([],tf.int64),
        # 'height':tf.FixedLenFeature([],tf.int64),
        # 'width':tf.FixedLenFeature([],tf.int64),
        # 'channel':tf.FixedLenFeature([],tf.int64),
    })

images=tf.decode_raw(features['image_raw'],tf.float32)
# heights = tf.cast(features['height'],tf.int32)
# widths = tf.cast(features['width'],tf.int32)
# channels = tf.cast(features['channel'],tf.int32)

#从原始图像数据解析出像素矩阵
labels = tf.cast(features['label'],tf.int64)
images = tf.reshape(images,[300,300,3])

#定义神经网络输入层图片大小
image_size=224
#预处理函数
distorted_image = preprocess_for_train(images,image_size,image_size,bbox=None)
#定义类别数
classes_num=30

#将处理后的图像和标签数据通过tf.train.shuffle_batch 整理成神经网络训练时需要的batch
min_after_dequeue = 1200
batch_size=64
capacity = min_after_dequeue+3*batch_size
image_batch,label_batch = tf.train.shuffle_batch([distorted_image,labels],batch_size=batch_size,
                                                 capacity=capacity,min_after_dequeue=min_after_dequeue,num_threads=15)

#定义神经网络的结构以及优化过程。image_batch 可以作为输入提供神经网络的输入层。
#label提供了输入batch中阳历的正确答案
with tf.name_scope('input'):
    keep_prob1=tf.placeholder("float")
#training
with resnet_inference.slim.arg_scope(resnet_inference.resnet_arg_scope(is_training=True)):
    net,end_point = resnet_inference.resnet_v2_50(inputs=image_batch,num_classes=classes_num,keep_prob=keep_prob1)
    net = tf.reshape(net, [-1, classes_num])
    loss = resnet_inference.loss(net=net,label_batch=label_batch)
    learning_rate = 0.001
    keep_prob=0.5
#优化器
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies([tf.group(*update_ops)]):
        train_step=optimizer.minimize(loss)
    predict_op=tf.argmax(net,1)
# train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss=loss)

#test and verfication
with resnet_inference.slim.arg_scope(resnet_inference.resnet_arg_scope(is_training=False)):
    net_test,end_point_test = resnet_inference.resnet_v2_50(inputs=image_batch,num_classes=classes_num,keep_prob=keep_prob1,reuse=True)
    net_test=tf.reshape(net_test,[-1,classes_num])
    predict_op_test = tf.argmax(net_test,1)

    correct_prediction = tf.equal(predict_op_test, label_batch)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#保存模型
ckpt_dir="./ckpt_dir"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)
#计数器变量，设置他的trainable=False,不需要被训练
global_step=tf.Variable(0,name='global_step',trainable=False)
saver=tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


with tf.Session(config=config) as sess:
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()

    #若训练中断，可从中断处开始训练
    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print(ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)  # restore all variables
    start=global_step.eval()
    print("start from",start)


    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)

    for i in range(start,300000):
        start_time=time.time()
        # print(i)
        _,loss_value=sess.run([train_step,loss],feed_dict={keep_prob1:keep_prob})
        global_step.assign(i).eval()
        duration =time.time()-start_time
        # print("step_time:",duration)
        if i%100==0:
            saver.save(sess,ckpt_dir+"/model.ckpt",global_step=global_step)
            accuracy_train=sess.run(accuracy,feed_dict={keep_prob1:1})
            print('step %d: loss = %.2f( %.3f sec) train_accuracy=%.3f'%(i,loss_value,duration,accuracy_train))

    coord.request_stop()
    coord.join()