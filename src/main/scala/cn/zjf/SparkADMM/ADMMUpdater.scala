package cn.zjf.SparkADMM

import breeze.linalg.{DenseVector, norm}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
trait ADMMUpdater{
  def xUpdate(data:Array[LabeledPoint],state: ADMMState): ADMMState
  def zUpdate(states: RDD[(Int,ADMMState)]): RDD[(Int,ADMMState)]
  def isStop(states: RDD[(Int,ADMMState)]):Boolean
  def uUpdate(state: ADMMState): ADMMState = {
    //state.copy( u = state.u + state.x - state.z)
    //println("uupdate")
    state.copy( u = state.u + state.x - state.z)
  }
}

object ADMMUpdater {
  def ADMMStop(rho:Double)(states: RDD[(Int,ADMMState)]):Boolean={
    val numrdd:Double = states.partitions.length
    val seqs = states.map(state => {
      val xNorm = norm(state._2.x)
      val uNorm = norm(state._2.u)
      val rNorm = norm(state._2.x - state._2.z)
      val sNorm = norm(state._2.z - state._2.z_pre)
      val zNorm = norm(state._2.z)
      DenseVector[Double](xNorm, uNorm, rNorm, sNorm,zNorm)
    }).reduce(_ + _)/numrdd
    val prires:Double = seqs(2)
    val duares:Double = seqs(3)
    val eps_pri:Double = math.sqrt(states.take(1)(0)._2.x.length)*1e-3+1e-3*math.max(seqs(0),seqs(4))
    val eps_dual:Double = math.sqrt(677399)*1e-3+1e-3*rho*seqs(1)
    println(prires+"\t"+eps_pri+"\t"+duares+"\t"+eps_dual)
    if(prires < eps_pri && duares < eps_dual)
      true
    else
      false
  }

  def linearZUpdate(lambda:Double,rho:Double)(states: RDD[(Int,ADMMState)])={
    val numStates = states.collect.length
    // TODO(tulloch) - is this epsilon > 0 a hack?
    //val epsilon = 0.00001 // avoid division by zero for shrinkage
    //同步所有的x和u
    val wBar = states.map(state=>state._2.x+state._2.u).reduce(_+_)
    //val xBar = average(states.map(_.x))/count
    //val xBar = states.map(_.x).reduce(_+_)./states.count()
    //val uBar = average(states.map(_.u))/count
    val s = rho /(rho * numStates + 2 * lambda)
    val zNew = s*wBar
    val newstates = states.map(state => {
      state._2.z_pre=state._2.z
      state._2.z = zNew
      state
    })
    newstates
  }
  def average(updates: RDD[DenseVector[Double]]): DenseVector[Double] = {
     updates.reduce(_ + _)
  }
}
