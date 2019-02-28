"""
A binary to train CIFAR-10 using multi_GPU in some devices.
Here we use 3 terminals to represent 2 works devices and 1 ps device
This code can only run on the terminal
Please use the following command to run the code:
    terminal 1： python cifar10_distribute_Train.py --job_name=ps --task_index=0
    terminal 2： python cifar10_distribute_Train.py --job_name=worker --task_index=1
    terminal 3： python cifar10_distribute_Train.py --job_name=worker --task_index=0
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
from datetime import datetime
import time

import tensorflow as tf

import cifar10
import cifar10_input

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_distribute_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('num_steps', 1,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_integer('log_frequency', 10,
                            """How often to log results to the console.""")

# 定义分布式参数
# 参数服务器parameter server节点
tf.app.flags.DEFINE_string('ps_hosts', 'localhost:2221',
                           'Comma-separated list of hostname:port pairs')
# 两个worker节点
tf.app.flags.DEFINE_string('worker_hosts', 'localhost:2222,localhost:2223',
                    'Comma-separated list of hostname:port pairs')
# 每台机器的 GPU 个数
tf.app.flags.DEFINE_integer("num_gpus", 0, "Total number of gpus for each machine."
                                           "If you don't use GPU, please set it to '0'")
# job_name和task_index通过终端输入获得
# 设置job name参数
tf.app.flags.DEFINE_string('job_name', None, 'job name: worker or ps')
# 设置任务的索引
tf.app.flags.DEFINE_integer('task_index', None, 'Index of task within the job')

# Global constants describing the CIFAR-10 data set.
IMAGE_SIZE = cifar10_input.IMAGE_SIZE
NUM_CLASSES = cifar10_input.NUM_CLASSES
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = cifar10_input.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

# Constants describing the training process.
MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
NUM_EPOCHS_PER_DECAY = 350.0  # Epochs after which learning rate decays.
LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.
INITIAL_LEARNING_RATE = 0.1  # Initial learning rate.


def train():
    """Train CIFAR-10 for a number of steps."""
    if FLAGS.job_name is None or FLAGS.job_name == '':
        raise ValueError('Must specify an explicit job_name !')
    else:
        print('job_name : %s' % FLAGS.job_name)
    if FLAGS.task_index is None or FLAGS.task_index == '':
        raise ValueError('Must specify an explicit task_index!')
    else:
        print('task_index : %d' % FLAGS.task_index)

    # 机器参数解析
    ps_spc = FLAGS.ps_hosts.split(',')
    worker_spec = FLAGS.worker_hosts.split(',')

    # 获取同步更新模型参数所需要的副本数
    num_workers = len(worker_spec)
    print("worker numbers:%d"%num_workers)
    # 创建集群
    cluster = tf.train.ClusterSpec({'ps': ps_spc,
                                    'worker': worker_spec})

    # 创建当前机器的server，用以连接到cluster
    # 这里flag通过接收终端输入的job_name和task_index来指定对应的设备
    server = tf.train.Server(
        cluster, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    # 如果当前节点是parameter server，则不再进行后续的操作，
    # 而是使用server.join等待worker工作
    if FLAGS.job_name == "ps":
        server.join()

    is_chief = (FLAGS.task_index == 0)

    if FLAGS.num_gpus > 0:
        # 避免gpu分配冲突：现在为相应机器中的每个worker分配task_num - > #gpu
        gpu = (FLAGS.task_index % FLAGS.num_gpus)
        worker_device = "/job:worker/task:%d/gpu:%d" % (FLAGS.task_index, gpu)
    elif FLAGS.num_gpus == 0:
        cpu = 0
        worker_device = "/job:worker/task:%d/cpu:%d" % (FLAGS.task_index, cpu)

    # 我们使用 tf.train.replica_device_setter 将涉及变量的操作分配到参数服务器上，并使用 CPU;
    # 将涉及非变量的操作分配到工作节点上，使用 worker_device 的值。
    # 在这个 with 语句之下定义的参数，会自动分配到参数服务器上去定义 ，
    # 如果有多个参数服务器，就轮流循环分配。
    # 在深度学习训练中，一般图的计算，对于每个worker task来说，都是相同的，
    # 所以我们会把所有图计算、变量定义等代码，都写到下面这个语句下。
    device_setter=tf.train.replica_device_setter(
                worker_device=worker_device,
                ps_device='/job:ps/cpu:0',
                cluster=cluster)
    with tf.device(device_setter):
        global_step = tf.train.get_or_create_global_step()

        # Get images and labels for CIFAR-10.
        images, labels = cifar10.distorted_inputs(FLAGS.batch_size)

        # Build a Graph that computes the logits predictions from the
        # inference model.
        logits = cifar10.inference(images)

        # Calculate loss.
        loss = cifar10.loss(logits, labels)
        num_batches_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size
        decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY)

        # Decay the learning rate exponentially based on the number of steps.
        lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE,
                                        global_step,
                                        decay_steps,
                                        LEARNING_RATE_DECAY_FACTOR,
                                        staircase=True)
        # 创建同步训练的优化器
        # tf.train.SyncReplicasOptimizer实质上是对原有优化器的一个扩展，
        # 我们传入原有优化器及其他参数，
        # 它会将原有优化器改造为同步分布式训练版本
        #opt = tf.train.SyncReplicasOptimizer(
        #    tf.train.GradientDescentOptimizer(lr),
        #    replicas_to_aggregate=num_workers,
        #    total_num_replicas=num_workers)
        # 记得传入global_step以同步
        opt=tf.train.GradientDescentOptimizer(lr)
        train_op = opt.minimize(loss, global_step=global_step)

        class _LoggerHook(tf.train.SessionRunHook):
            """Logs loss and runtime."""
            def begin(self):
                self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(loss)  # Asks for loss value.

            def after_run(self, run_context, run_values):
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self.start_time = current_time

                    loss_value = run_values.results
                    examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    #print(format_str % (datetime.now(), self._step, loss_value,
                    #                    examples_per_sec, sec_per_batch))
        hooks = [#opt.make_session_run_hook(is_chief),
                 tf.train.StopAtStepHook(num_steps=FLAGS.max_steps),
                 tf.train.NanTensorHook(loss),
                 _LoggerHook()]

        sess_config = tf.ConfigProto(
            # 软放置
            # 如果该操作函数没有 GPU 实现时
            # 会自动使用 CPU 设备
            allow_soft_placement=True,
            # 是否打印操作所对应的执行设备的信息
            log_device_placement=False,
            # device_filters:硬件过滤器
            # 如果被设置的话，会话会忽略掉所有不匹配过滤器的硬件。
            device_filters=["/job:ps",
                            "/job:worker/task:%d" % FLAGS.task_index])

        with tf.train.MonitoredTrainingSession(
                master=server.target,
                is_chief=is_chief,
                checkpoint_dir=FLAGS.train_dir,
                save_checkpoint_steps=1,
                hooks=hooks,
                config=sess_config) as mon_sess:
            while not mon_sess.should_stop():
                print(mon_sess.run([global_step, loss]))
                mon_sess.run(train_op)


def main(argv=None):
    # pylint: disable=unused-argument
    cifar10.maybe_download_and_extract()
    train()


if __name__ == '__main__':
  tf.app.run()