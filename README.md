# ARCore NanoDet Object & Depth Detection

An advanced, real-time Android machine learning application that pairs [Tencent's NanoDet](https://github.com/RangiLyu/nanodet) object detection architecture with Google's **ARCore Depth API**. 

This application seamlessly overlays ultra-fast object detection bounding boxes with real-world physical spatial measurements (meters) using Google's native AR hardware sensors.

---

## 📸 Features

*   **⚡ Real-Time Object Detection**: Uses the highly efficient NanoDet architecture powered by the `ncnn` framework to detect up to 80 different COCO classes with near-zero latency.
*   **📏 Automatic Depth Estimation**: Automatically integrates with Google ARCore to extract physical distance mapping data (Z-Depth) corresponding directly to the bounding box of the detected object.
*   **🎨 Glassmorphic Material UI**: A fully custom, sleek, futuristic user interface with an edge-to-edge AR camera viewfinder, glowing bounding boxes, and stylized control cards.
*   **⚙️ Hardware Acceleration**: Seamlessly toggle between running inference algorithms on your CPU natively or farming the logic out to your GPU via Vulkan for maximized frame rates!
*   **🔌 Modular Model Switching**: Change NanoDet model architectures dynamically on-the-fly (`m`, `m-416`, `ELC-half`, etc.) without needing a hard app restart.
*   **🕶️ Splash Screen Integration**: Boot into a stylized splash screen while the neural networks buffer into device memory automatically.

---

## DEMO
----
https://github.com/user-attachments/assets/e1f034d2-01e4-40d0-b977-db27c0839bb6



## 🛠️ Architecture and Stack

*   **Language**: Java (UI / ARCore interfacing), C++ (JNI / ncnn networking logic)
*   **Android SDK**: Target 35 / Min 24
*   **Deep Learning Framework**: [ncnn](https://github.com/Tencent/ncnn) (High-performance neural network inference computing framework optimized for mobile platforms)
*   **AR Framework**: Google Play Services for AR (ARCore)
*   **Native Build System**: CMake

---

## 🧩 Building the App

This app requires the Android NDK to compile the `ncnn` C++ inference logic. 

### Prerequisites
1.  **Android Studio** installed and updated.
2.  **Android NDK** and **CMake** installed via the Android Studio SDK Manager.
    > The app currently targets NDK version `29.0.14206865`. If you do not have this installed, navigate to the `app/build.gradle` file and update the `ndkVersion` field to match an NDK version you actively possess, or download version 29 via the SDK manager.
3.  A physical Android device that officially supports **ARCore Depth API** (Emulators will not easily process physical AR hardware demands).

### Compilation
1. Clone this repository and open the root `ncnn-android-nanodet-master` folder in Android Studio.
2. Allow Gradle to perform its initial sync to pull down ARCore and Material Design dependencies.
3. Build the project (`Build -> Make Project`). Gradle will automatically farm out C++ logic compilation to CMake via the NDK to create `libnanodetncnn.so`.
4. Deploy the APK to your physical device.

---

## 📝 Usage

Upon starting the App:
1. Accept the **Camera Permissions** prompt so ARCore can begin mapping your environment.
2. **Aim** your physical camera at common objects (People, cars, cups, dogs, lapops, etc.). The app will automatically draw bounded boxes.
3. Look at the label hovering above the box. It will read out the detected class and the exact physical distance (e.g., `person 1.45 m`).
4. **Radio Buttons**: Toggle between **"Obj. Detect + Depth"** and **"Obj. Detect Only"** to turn off the laser depth-sensing engine in ARCore to boost raw detection frame rates if you solely care about what an object is rather than where it is.
5. **Dropdowns**: Use the Dropdowns to change network modes or swap computational logic out to your hardware GPU. 

---

## ⚠️ Notes 

* **Camera Restriction**: This application strictly utilizes the rear-facing tracking camera of your device. The *front-facing* camera lacks the stereoscopic/ToF hardware sensors required to build accurate ARCore depth maps on a vast majority of Android devices. 
* **Model Assets**: The included `.bin` and `.param` files located in `app/src/main/assets` govern the neural net matrices. If you train a custom NanoDet model, dump your exported `.param` and `.bin` weights into this directory and register them in the Java / C++ arrays.

---

*This project is built upon the foundational work provided by the official Tencent NCNN architecture examples and RangiLyu's NanoDet structure.*
