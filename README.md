# Magic data analysis


### Description

The data are Monte Carlo generated to simulate registration of high energy gamma particles in a ground-based atmospheric Cherenkov gamma telescope using the imaging technique. Cherenkov gamma telescope observes high energy gamma rays, taking advantage of the radiation emitted by charged particles produced inside the electromagnetic showers initiated by the gammas, and developing in the atmosphere. This Cherenkov radiation (of visible to UV wavelengths) leaks through the atmosphere and gets recorded in the detector, allowing reconstruction of the shower parameters. The available information consists of pulses left by the incoming Cherenkov photons on the photomultiplier tubes, arranged in a plane, the camera. Depending on the energy of the primary gamma, a total of few hundreds to some 10000 Cherenkov photons get collected, in patterns (called the shower image), allowing to discriminate statistically those caused by primary gammas (signal) from the images of hadronic showers initiated by cosmic rays in the upper atmosphere (background). 

Typically, the image of a shower after some pre-processing is an elongated cluster. Its long axis is oriented towards the camera center if the shower axis is parallel to the telescope's optical axis, i.e. if the telescope axis is directed towards a point source. A principal component analysis is performed in the camera plane, which results in a correlation axis and defines an ellipse. If the depositions were distributed as a bivariate Gaussian, this would be an equidensity ellipse. The characteristic parameters of this ellipse (often called Hillas parameters) are among the image parameters that can be used for discrimination. The energy depositions are typically asymmetric along the major axis, and this asymmetry can also be used in discrimination. There are, in addition, further discriminating characteristics, like the extent of the cluster in the image plane, or the total sum of depositions. 

The program was run with parameters allowing to observe events with energies down to below 50 GeV.

### Dataset

The dataset is available at this [link](https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data)

Attribute Information:

1. fLength: continuous # major axis of ellipse [mm] 
2. fWidth: continuous # minor axis of ellipse [mm] 
3. fSize: continuous # 10-log of sum of content of all pixels [in #phot] 
4. fConc: continuous # ratio of sum of two highest pixels over fSize [ratio] 
5. fConc1: continuous # ratio of highest pixel over fSize [ratio] 
6. fAsym: continuous # distance from highest pixel to center, projected onto major axis [mm] 
7. fM3Long: continuous # 3rd root of third moment along major axis [mm] 
8. fM3Trans: continuous # 3rd root of third moment along minor axis [mm] 
9. fAlpha: continuous # angle of major axis with vector to origin [deg] 
10. fDist: continuous # distance from origin to center of ellipse [mm] 
11. class: g,h # gamma (signal), hadron (background) 

g = gamma (signal): 12332 
h = hadron (background): 6688 

For technical reasons, the number of h events is underestimated. In the real data, the h class represents the majority of the events. 

The simple classification accuracy is not meaningful for this data, since classifying a background event as signal is worse than classifying a signal event as background. For comparison of different classifiers an ROC curve has to be used. The relevant points on this curve are those, where the probability of accepting a background event as signal is below one of the following thresholds: 0.01, 0.02, 0.05, 0.1, 0.2 depending on the required quality of the sample of the accepted events for different experiments.


### References

Bock, R.K., Chilingarian, A., Gaug, M., Hakl, F., Hengstebeck, T., Jirina, M., Klaschka, J., Kotrc, E., Savicky, P., Towers, S., Vaicilius, A., Wittek W. (2004). 
Methods for multidimensional event classification: a case study using images from a Cherenkov gamma-ray telescope. 
Nucl.Instr.Meth. A, 516, pp. 511-528. 

P. Savicky, E. Kotrc. 
Experimental Study of Leaf Confidences for Random Forest. 
Proceedings of COMPSTAT 2004, In: Computational Statistics. (Ed.: Antoch J.) - Heidelberg, Physica Verlag 2004, pp. 1767-1774. 

J. Dvorak, P. Savicky. 
Softening Splits in Decision Trees Using Simulated Annealing. 
Proceedings of ICANNGA 2007, Warsaw, (Ed.: Beliczynski et. al), Part I, LNCS 4431, pp. 721-729.


### Assignments

The main goal is to distinguish signal and background events. Two approaches can be followed: 1) exploiting the physics of the detection principle 2) use a physics-agnostic multivariate technique, e.g. a neural network.

* Study the features of the datasets and compare them for signal and background events
* Study the correlations among the features of the datasets for signal and background events
* Compute the "mean-scaled-width" and the "mean-scale-length", i.e. rescale by means of their mean and standard deviation the "Width" and "Length" distributions. Compare them for signal and background events in the cases of little or a lot of light ("fSize") 
* Perform a Principal Component Analysis on that dataset for the signal and the background events
* Perform a multivariate analysis with the technique you prefer and evaluate its performance (e.g. in terms of Area Under the (ROC) Curve)

----------
==========Update==========
----------
# CVAE + CGaN


### **Hybrid Model Overview**

The hybrid model combines two generative architectures: **Conditional Variational Autoencoder (CVAE)** and **Conditional Generative Adversarial Network (CGAN)**. These components work together to analyze, generate, and classify **Cherenkov radiation image data** from the MAGIC telescope. The goal is to distinguish gamma-ray signals (g) from hadronic showers (h) using both physics-based reconstruction and physics-agnostic multivariate analysis.

---

### **1. Why a Hybrid Model?**

Cherenkov radiation images are complex, high-dimensional, and structured. A hybrid model leverages the unique strengths of both CVAE and CGAN:

1. **CVAE**:
   - Learns a probabilistic latent representation of the data.
   - Captures the underlying structure and relationships between image features and auxiliary physical conditions (e.g., particle momentum, position).
   - Can reconstruct original data, useful for interpreting learned features.

2. **CGAN**:
   - Learns to generate realistic images by combining random noise with auxiliary conditions.
   - Improves the quality of generated samples through adversarial training.
   - Helps distinguish between real and synthetic Cherenkov images.

By combining these, the hybrid model:
- Encodes images into a **latent space** for interpretation (CVAE).
- Generates synthetic images for augmentation and modeling (CGAN).
- Provides a discriminator for distinguishing between real and synthetic images.

---

### **2. CVAE (Conditional Variational Autoencoder)**

#### **Purpose**:
- To model the distribution of Cherenkov radiation images and reconstruct them.
- To learn a **latent space** where similar features are grouped, enabling physics-informed analysis.

#### **Components**:
1. **Encoder**:
   - Maps the input image and condition (e.g., particle metadata) to a latent distribution defined by `z_mean` and `z_log_var`.
   - Outputs a probabilistic latent space, ensuring variability in reconstructions.

2. **Latent Space Sampling**:
   - A random sample is drawn from the learned latent distribution.
   - Allows variability in reconstructions, modeling the natural variations in Cherenkov images.

3. **Decoder**:
   - Reconstructs the original image from the sampled latent vector and condition.
   - Ensures that the latent representation retains enough information to recreate the input.

---

### **3. CGAN (Conditional Generative Adversarial Network)**

#### **Purpose**:
- To generate **realistic Cherenkov images** conditioned on auxiliary features (e.g., particle energy, position).
- To differentiate between real (from data) and synthetic (from the generator) images, refining the generator's learning.

#### **Components**:
1. **Generator**:
   - Takes a random noise vector and auxiliary conditions as input.
   - Produces synthetic Cherenkov images by learning patterns in the data.
   - Mimics the characteristics of real gamma-ray and hadronic showers.

2. **Discriminator**:
   - Takes an image (real or synthetic) and auxiliary conditions as input.
   - Outputs a probability indicating whether the image is real or fake.
   - Provides feedback to the generator to improve its ability to create realistic images.

---

### **4. How the Hybrid Model Works**

The hybrid model integrates the CVAE and CGAN components, creating a unified framework for both data reconstruction and generation:

1. **Encoding (CVAE)**:
   - Input Cherenkov images are encoded into a latent space that captures meaningful physical and statistical features.

2. **Decoding (CVAE)**:
   - The latent representation, combined with auxiliary conditions, is decoded back into the original image space.
   - This ensures the latent space retains enough information to recreate the original data.

3. **Image Generation (CGAN)**:
   - The generator uses random noise and auxiliary conditions to create synthetic images.
   - This allows the model to simulate realistic Cherenkov showers, useful for data augmentation or understanding the generative process.

4. **Discrimination (CGAN)**:
   - The discriminator evaluates the quality of the generated images, guiding the generator to improve.

---

### **5. Applications of the Hybrid Model**

1. **Physics-Informed Reconstruction**:
   - The CVAE’s latent space provides a compact representation of the data, enabling feature extraction and interpretation.
   - Helps identify key physical characteristics that distinguish gamma-ray signals from hadronic showers.

2. **Data Augmentation**:
   - The CGAN generates synthetic Cherenkov images, augmenting the dataset to improve classification performance.

3. **Classification**:
   - The discriminator can act as a binary classifier, distinguishing between gamma-ray signals (g) and hadronic showers (h).

4. **Feature Exploration**:
   - The latent space captures the relationships between features, providing insights into the data’s structure.

5. **Error Analysis**:
   - By analyzing incorrectly classified events, researchers can identify overlapping features or limitations in the dataset.

---

### **6. Benefits of the Hybrid Approach**

1. **Data Representation**:
   - The CVAE provides a probabilistic latent space, ensuring variability in reconstructions.
   - The CGAN refines the generator’s ability to model realistic data.

2. **Versatility**:
   - Combines physics-informed (CVAE) and physics-agnostic (CGAN) approaches.
   - Supports both data reconstruction and synthetic data generation.

3. **Improved Performance**:
   - The hybrid model leverages complementary strengths of CVAE and CGAN, improving overall modeling and classification accuracy.

---

### **7. Challenges and Considerations**

1. **Data Imbalance**:
   - Hadronic showers (h) are underrepresented in the dataset. Proper handling, such as rebalancing or weighting, is critical.

2. **Hyperparameter Tuning**:
   - Key parameters like the latent dimension, batch size, and learning rates need careful optimization.

3. **Training Complexity**:
   - Adversarial training (CGAN) requires balancing the generator and discriminator, which can be challenging.

4. **Interpretability**:
   - While the CVAE provides some interpretability via the latent space, the CGAN’s outputs may require further analysis to ensure physical relevance.

---

### **8. Summary of the Hybrid Model**

| **Component**      | **Purpose**                          | **Key Functionality**                                                                                  |
|---------------------|--------------------------------------|--------------------------------------------------------------------------------------------------------|
| **CVAE Encoder**    | Data compression                    | Encodes Cherenkov images into a compact latent space conditioned on auxiliary features.               |
| **CVAE Decoder**    | Data reconstruction                 | Reconstructs images from the latent space and auxiliary features, ensuring retention of key details.  |
| **CGAN Generator**  | Data generation                     | Produces synthetic Cherenkov images by learning patterns from real data.                              |
| **CGAN Discriminator** | Real-vs-fake classification       | Differentiates between real and generated images, refining the generator’s learning.                  |

This hybrid approach is well-suited for distinguishing gamma-ray signals from hadronic showers, providing both a robust classification framework and insights into the underlying data structure.


### References

 - Fedor Sergeev et al 2021 J. Phys.: Conf. Ser. 1740 012028 [https://doi.org/10.1088/1742-6596/1740/1/012028]

 - Chisholm, Andrew, et al. "Non-parametric data-driven background modelling using conditional probabilities." Journal of High Energy Physics 2022.10 (2022): 1-34.
 [https://doi.org/10.48550/arXiv.2112.00650]

