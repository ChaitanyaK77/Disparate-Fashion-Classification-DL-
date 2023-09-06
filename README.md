# Disparate-Fashion-Classification-DL-

The Disparate Fashion Classification project is a deep learning endeavor that harnesses the power of TensorFlow to perform accurate and efficient classification of fashion items using the Fashion MNIST dataset. This dataset comprises a diverse range of grayscale images, each depicting clothing and accessories across ten distinct categories, making it a valuable resource for training and testing machine learning models. The project offers a user-friendly pipeline that accommodates both model training and inference, providing flexibility for users with varying levels of expertise. Users can choose to embark on model training from scratch, tailoring the architecture to their specific needs, or alternatively, employ pre-trained models for swift and reliable fashion item classification. By harnessing the capabilities of convolutional neural networks (CNNs), the project aims to enhance classification accuracy and generalization, making it an invaluable tool for various applications within the fashion industry and computer vision research. The project welcomes contributions from the open-source community, fostering collaboration to advance its capabilities and ensure its continued relevance in the ever-evolving field of deep learning and fashion analysis.



## Getting Started

1. **Clone the Repository**: 
   ```sh
   git clone https://github.com/your-username/disparate-fashion-classification.git
   ```

2. **Install Dependencies**: 
   ```sh
   pip install -r requirements.txt
   ```

## Usage

- **Inference**: Use a pre-trained model for image classification.
  ```sh
  python inference.py --model_path models/pretrained_model.h5 --image_path path_to_image.jpg
  ```

- **Training**: Train your own model.
  ```sh
  python train.py --model_name my_model --epochs 5
  ```


## Evaluation

Evaluate model performance:
```sh
python evaluate.py --model_path path_to_trained_model.h5
```



## Acknowledgements

- Fashion MNIST Dataset: [GitHub Repo](https://github.com/zalandoresearch/fashion-mnist)
- TensorFlow: [Website](https://www.tensorflow.org/)
- NumPy: [Website](https://numpy.org/)
- Matplotlib: [Website](https://matplotlib.org/)

--- 

