import org.apache.spark.mllib.linalg.{Vectors,SparseVector}
import breeze.linalg
import cn.zjf.SparkADMM.MyBLAS._

object test {
  def main(args: Array[String]): Unit = {
    val arr1 = Array[Double](1,0,0,0,5)
    val arr2 = Array[Double](1,1,1,1,1)
    val sparse = Vectors.sparse(5,Seq((0,1.0),(4,5.0)))
    val dense = linalg.DenseVector(arr2)
    var tmp:Double = 1;
    val aaa = dense.foreach(x=> {
      tmp+=x
    }
    )
    println(aaa)
    println("tmp is:"+tmp)
    val res = dot(sparse.toSparse,arr2)
    axpy(1,sparse.toSparse,dense.data)
    println(dense)
    println(res.toString)
  }

}
