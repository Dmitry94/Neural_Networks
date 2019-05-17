# Hand-written models.

In this folder hand-written models are stored. They were implemented in 2016.
Their implementation is based on cs-231 course materials.

## Results

* **Nearest Neighbor classifier**
  
  It was tested on CIFAR-10 dataset. With *k=1* I got the following result:
  
    ```
    Training time = 0.003066000000000013
    Prediction time = 553.881721
    Got accuracy = 0.2176 at test data.
    ```
  
* **SVM classifier**

    ```
    SVM training time = 37.834251 (10000 iters)
    SVM predicting time = 0.6280599999999978
    SVM accuracy = 0.2666
    ```

* **Softmax classifier**

    ```
    Softmax training time = 53.29244 (10000 iters)
    Softmax predicting time = 0.665188999999998
    Softmax accuracy = 0.2821
    ```

* **Neural network classifier**

    ```
    NeuralNet training time = 204.381503 (10000 iters, 100-150-100 arch)
    NeuralNet predicting time = 1.0823270000000207
    NeuralNet accuracy = 0.4516
    ```
    
No specific hyperparams were applied. All params can be found at scripts.