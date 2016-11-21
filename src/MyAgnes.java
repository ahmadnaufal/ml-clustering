import weka.clusterers.AbstractClusterer;
import weka.core.Instances;

import java.util.ArrayList;

/**
 * Created by Ahmad on 11/20/2016.
 */
public class MyAgnes extends AbstractClusterer {

    Instances mInstances;
    int nClusters;

    public MyAgnes() {
        nClusters = 2;
    }

    public MyAgnes(int clusters) {
        nClusters = clusters;
    }

    @Override
    public void buildClusterer(Instances instances) throws Exception {
        mInstances = instances;
        int n = mInstances.numInstances();
        if (n < 1) {
            System.out.println("No input data from the instances.");
            return;
        }

        ArrayList<Integer> [] dataClusters = new ArrayList[n];
        for (int i = 0; i < n; ++i) {
            // assign all instance index to all clusters
            dataClusters[i] = new ArrayList<>();
            dataClusters[i].add(i);
        }

        // join nodes based on euclidean distances

    }

    @Override
    public int numberOfClusters() throws Exception {
        return 0;
    }
}
