# ConvMixer

Implementation of ConvMixer in TensorFlow and Keras

<a href="https://colab.research.google.com/drive/1m-faU1DmBZlqkVY_tcYnOcepGlOyJ5K9?usp=sharing" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

**Paper Link:** https://openreview.net/pdf?id=TVHS5Y4dNvM <br>
Note: Paper is under review for ICLR 2022

## Model Architechture

![](https://i.imgur.com/Yd7gpMP.png) 

## Installation

```
pip install -q tensorflow-addons
```
Note: We are using TensorFlow-Addons for using the `AdamW` optimiser and `GeLU` activation function.

## Results

![Unknown-2](https://user-images.githubusercontent.com/41967348/137559060-96c6c84a-7055-4f3d-ade1-415e5a756880.png) ![Unknown](https://user-images.githubusercontent.com/41967348/137559078-0f095bd4-e119-457c-ac79-7caa5e9a076e.png)

> TensorBoard Link: https://tensorboard.dev/experiment/bkhqOz0RQ1Cv5dwrDQySMQ/

Note: Trained `25 Epochs` and got a top-5-accuracy of 64.41%

## Future Work

* To train on 150 epochs
* To train model on ImageNet dataset

## License
```
MIT License

Copyright (c) 2021 Sayan Nath

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
