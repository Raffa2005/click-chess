import unittest
import numpy as np
from tinychess.crossentropy import losses, empirical_floor, fit_temperature


class CrossEntropyTests(unittest.TestCase):
    def test_mask_and_extreme_logits_without_clipping(self):
        logits=np.array([[0.,0.,-np.inf],[1000.,0.,-np.inf]])
        np.testing.assert_allclose(losses(logits,[0,1]),[np.log(2),1000.])

    def test_frequency_floor(self):
        r=empirical_floor(['a','a','a','a','b'],[0,0,1,1,0])
        self.assertAlmostEqual(r['empirical_lookup_minimum_nats'],4*np.log(2)/5)
        self.assertEqual(r['singleton_decisions'],1)
        self.assertEqual(r['conflicting_label_decisions'],4)

    def test_calibration_recovers_known_distribution(self):
        scores=np.tile([np.log(3)*2,0.],(400,1))
        labels=np.array([0]*300+[1]*100)
        self.assertAlmostEqual(fit_temperature(scores,labels),2.,places=5)


if __name__=='__main__':unittest.main()
