import weka.clusterers.AbstractClusterer;
import weka.core.EuclideanDistance;
import weka.core.Instances;

import java.util.ArrayList;

/**
 * Created by Ahmad on 11/20/2016.
 */
public class MyAgnes extends AbstractClusterer {

    private Instances mInstances;
    private int mNumClusters;
    EuclideanDistance mDistanceFunction = new EuclideanDistance();

    public MyAgnes() {
        mNumClusters = 2;
    }

    public MyAgnes(int clusters) {
        mNumClusters = clusters;
    }

    @Override
    public void buildClusterer(Instances instances) throws Exception {
        mInstances = instances;
        if (mInstances.numInstances() < 1) {
            System.out.println("No input data from the instances.");
            return;
        }

        int nClusters = mInstances.numInstances();
        ArrayList<Integer> [] dataClusters = new ArrayList[nClusters];
        for (int i = 0; i < nClusters; ++i) {
            // assign all instance index to all clusters
            dataClusters[i] = new ArrayList<>();
            dataClusters[i].add(i);
        }

        // join nodes based on euclidean distances
        double[][] initDistances = initDistancesPerInstance(nClusters, dataClusters);
        double[][] distances = initDistances.clone();

        // start merging clusters
        while (mNumClusters < nClusters) {
            int iMin = -1, jMin = -1;
            double minDistance = Double.MAX_VALUE;

            for (int i = 0; i < mInstances.numInstances(); ++i) {
                if (dataClusters[i].size() > 0) {
                    for (int j = i+1; j < mInstances.numInstances(); ++i) {
                        if (dataClusters[i].size() > 0) {
                            if (initDistances[i][j] < minDistance) {
                                minDistance = initDistances[i][j];
                                iMin = i;
                                jMin = j;
                            }
                        }
                    }
                }
            }

            dataClusters[iMin].addAll(dataClusters[jMin]);
            dataClusters[jMin].clear();

            for (int i = 0; i < mInstances.numInstances(); ++i) {
                if (i != iMin && dataClusters[i].size() > 0) {
                    int i1 = Math.min(iMin, i);
                    int i2 = Math.max(iMin, i);
                    double distance = getClusterDistance(initDistances, dataClusters[i1], dataClusters[i2]);
                    distances[i1][i2] = distance;
                }
            }

            nClusters--;
        }
    }

    private double[][] initDistancesPerInstance(int numClusters, ArrayList<Integer>[] dataClusters) {
        double[][] distanceMatrix = new double[numClusters][numClusters];
        for (int i = 0; i < numClusters; ++i) {
            distanceMatrix[i][i] = 0;
            for (int j = i+1; j < numClusters; ++j) {
                distanceMatrix[i][j] = getInitClusterDistance(dataClusters[i], dataClusters[j]);
                // distanceMatrix[j][i] = distanceMatrix[i][j];
            }
        }

        return distanceMatrix;
    }

    private double getClusterDistance(double [][] initDistance, ArrayList<Integer> cluster1, ArrayList<Integer> cluster2) {
        // we are using single link
        // find the minimum link
        double minDistance = Double.MAX_VALUE;
        for (int i = 0; i < cluster1.size(); ++i) {
            for (int j = 0; j < cluster2.size(); ++j) {
                if (initDistance[i][j] < minDistance)
                    minDistance = initDistance[i][j];
            }
        }

        return minDistance;
    }

    private double getInitClusterDistance(ArrayList<Integer> cluster1, ArrayList<Integer> cluster2) {
        return mDistanceFunction.distance(mInstances.instance(cluster1.get(0)), mInstances.instance(cluster2.get(0)));
    }

    @Override
    public int numberOfClusters() throws Exception {
        return Math.min(mInstances.numInstances(), mNumClusters);
    }
}
