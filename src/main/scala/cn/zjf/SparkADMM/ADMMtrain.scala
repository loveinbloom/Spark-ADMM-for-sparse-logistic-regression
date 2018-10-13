package cn.zjf.SparkADMM

import org.apache.spark.mllib.regression.LabeledPoint
import MyLoadLibSVMFile._
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object ADMMtrain {
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf()
    conf.setAppName("SparkADMM")
    //conf.setMaster("local[8]")
    val sc = new SparkContext(conf)
    sc.setLogLevel("ERROR")
    //println("partion:"+sc.defaultMinPartitions)
    val trainingdata:RDD[LabeledPoint] = loadLibSVMFile(sc,"hdfs://node0:9000/data/rcv1_test.binary",-1,8)
    SparseLogisticRegressionADMM.train(trainingdata,100,1.0,1.0)
  }
}
