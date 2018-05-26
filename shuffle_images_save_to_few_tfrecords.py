import tensorflow as tf
import data as data
import matplotlib.pyplot as plt

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
#生成字符串型
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

images,labels = data.load_data(dirname="./object_detection_api", one_hot=False, resize_pics=(300, 300))
example_num=len(images)
files_num=10
sample_num_per_files=int(example_num/files_num)

index=0
for i in range(files_num):
    filename=("./shuffle_data_few_tfrecords/data.tfrecords-%.5d-of-%.5d"%(i,files_num))
    writer = tf.python_io.TFRecordWriter(filename)
    for j in range(sample_num_per_files):
        print(labels[index])
        # plt.imshow(images[index])
        # plt.show()
        img_raw=images[index].tostring()
        example=tf.train.Example(features=tf.train.Features(feature={
                'image_raw':_bytes_feature(img_raw),
                'label':_int64_feature(labels[index]),
            }))
        writer.write(example.SerializeToString())
        index=index+1
    writer.close()

