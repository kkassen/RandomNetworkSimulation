"""
Modifications: Kyle Kassen, Tate Teague
Dataset Source: https://networkrepository.com/rec-amazon.php
Cite: Professor John McDonald, DePaul University
      Annotated Algorithms in Python By Massimo Di Pierro
"""
import numpy as np

def bootstrap(x, confidence=.95, nSamples=100):
    means = []
    for k in range(nSamples):
        sample = np.random.choice(x, size=len(x), replace=True)
        means.append(np.mean(sample))
        means.sort()
        leftTail = int(((1.0 - confidence)/2) * nSamples)
        rightTail = (nSamples - 1) - leftTail
    return means[leftTail], np.mean(x), means[rightTail]

class MonteCarlo:

    def SimulateOnce(self):
        raise NotImplementedError

    def SimulateOneBatch(self, batchSize):
        raise NotImplementedError

    def var(self, risk = .05):
        if hasattr(self, "results"): 
    	    self.results.sort()
    	    index = int(len(self.results)*risk)   
    	    return(self.results[index]) 
        else:
            print("RunSimulation must be executed before the method 'var'")
            return 0.0

    def RunSimulation(self, threshold=.001, simCount=100000):
        self.results = []
        sum1 = 0.0
        sum2 = 0.0

        self.convergence = False
        for k in range(1, simCount + 1):
            x = self.SimulateOnce()

            # we added this outer loop to iterate over our list of metrics passed from the SimulateOnce() function
            for i in range(len(x)):
                self.results.append(x)
                sum1 += x[i]
                sum2 += x[i]*x[i]

                if k > 100:
                    mu = float(sum1)/k
                    var = (float(sum2)/(k-1)) - mu*mu
                    dmu = np.sqrt(var / k)

                    if dmu < abs(mu) * threshold:
                        self.convergence = True

        # Creating lists to hold the results for each of our metrics
        degree = []
        betweenness = []
        connectedness = []

        # Looping through our results and organizing by metric
        for i in range(len(self.results)):
            degree.append(self.results[i][0])
            betweenness.append(self.results[i][1])
            connectedness.append(self.results[i][2])

        # Calling bootstrap on each metric and returning our tuple of tuples
        return bootstrap(degree), bootstrap(betweenness), bootstrap(connectedness)

    def RunBatchedSimulation(self, batchSize=100, threshold=.001, simCount=100000):
        self.results = np.array([])
        sum1 = 0.0
        sum2 = 0.0
        

        self.convergence = False
        for k in range(1, simCount + 1, batchSize):
            x = self.SimulateOneBatch(batchSize)
            self.results = np.append(self.results, x)
            sum1 += np.sum(x)
            sum2 += np.sum(x*x)

            if k > 100:
                nSamples = k * batchSize
                mu = float(sum1)/nSamples
                var = (float(sum2)/(nSamples-1)) - mu*mu
                dmu = np.sqrt(var / nSamples)

                if dmu < abs(mu) * threshold:
                    self.convergence = True

        return bootstrap(self.results)


            