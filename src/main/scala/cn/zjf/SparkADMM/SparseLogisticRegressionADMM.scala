package cn.zjf.SparkADMM

import breeze.linalg.{DenseVector,  norm}
import breeze.optimize.{DiffFunction, LBFGS}
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.linalg
import org.apache.spark.mllib.optimization.Optimizer
import org.apache.spark.mllib.regression.{GeneralizedLinearAlgorithm, LabeledPoint}
import org.apache.spark.mllib.util.DataValidators
import org.apache.spark.rdd.RDD
import MyBLAS._

case class SparseLRADMMPrimalUpdater(lambda:Double, rho:Double,
                                     lbfgsMaxNumIterations: Int = 5,
                                     lbfgsHistory: Int = 3,
                                     lbfgsTolerance: Double = 1E-4)
  extends ADMMUpdater{
  override def xUpdate(data:Array[LabeledPoint],state: ADMMState): ADMMState = {
    val f = new DiffFunction[DenseVector[Double]] {
      override def calculate(x: DenseVector[Double]) = {
        (objective(data,state)(x), gradient(data,state)(x))
      }
    }
    val lbfgs = new LBFGS[DenseVector[Double]](maxIter = lbfgsMaxNumIterations,
      m = lbfgsHistory,
      tolerance = lbfgsTolerance)
    //warm start
    val xNew = lbfgs.minimize(f,state.x)
    //println("x_update")
    state.copy(x=xNew)
  }
  override def zUpdate(states:RDD[(Int,ADMMState)]): RDD[(Int,ADMMState)] = {
    ADMMUpdater.linearZUpdate(lambda = lambda, rho = rho)(states)
  }
  override def isStop(states: RDD[(Int,ADMMState)]): Boolean = {
    ADMMUpdater.ADMMStop(rho=rho)(states)
  }
  def objective(data:Array[LabeledPoint],state:ADMMState)(weight:DenseVector[Double]):Double={
    val lossObjective = data
      .map(lp=>{
        //val margin = lp.label * (weight dot SparseVector[Double](lp.features.toArray))
        val margin = lp.label*dot(lp.features.toSparse,weight.data)
        logPhi(margin)
      }).sum
    val regularizerObjective = norm(weight-state.z+state.u)
    val totalObjective =  rho/2*regularizerObjective*regularizerObjective + lossObjective
    //println("object value"+totalObjective)
    totalObjective
  }
  //对于每个RDD中的features求梯度
  def gradient(data:Array[LabeledPoint],state: ADMMState)(weights:DenseVector[Double]):DenseVector[Double]={
    var lossGradient:DenseVector[Double] = DenseVector.zeros(weights.length)
    data.foreach(lp=>{
      val margin = lp.label * dot(lp.features.toSparse,weights.data)
      val a = lp.label * (phi(margin) - 1)
      axpy(a,lp.features.toSparse,lossGradient.data)
    }
    )
    val regularizerGradient = (weights - state.z + state.u)
    val totalGradient = lossGradient + rho * regularizerGradient
    totalGradient

  }

  private def logPhi(margin: Double): Double = {
    // TODO(tulloch) - do we need to clamp here?
    if(margin > 0)
      math.log(1+math.exp(-margin))
    else
      (-margin+math.log(1+math.exp(margin)))
  }

  private def phi(margin: Double): Double = {
    // TODO(tulloch) - do we need to clamp here?
    1/(1+math.exp(-margin))
  }

}

class SparseLogisticRegressionADMM(numIterations:Int,lambda:Double,rho:Double)
  extends GeneralizedLinearAlgorithm[LogisticRegressionModel] with Serializable {
  override def optimizer: Optimizer = new ADMMOptimizer(
    numIterations,
    new SparseLRADMMPrimalUpdater(lambda = lambda, rho = rho))
  override val validators = List(DataValidators.binaryLabelValidator)
  override protected def createModel(weights: linalg.Vector, intercept: Double): LogisticRegressionModel
  = new LogisticRegressionModel(weights, intercept)
}
object SparseLogisticRegressionADMM{
  def train(
             input: RDD[LabeledPoint],
             numIterations: Int,
             lambda: Double,
             rho: Double) = {
    new SparseLogisticRegressionADMM(numIterations, lambda, rho).run(input)
  }
}
