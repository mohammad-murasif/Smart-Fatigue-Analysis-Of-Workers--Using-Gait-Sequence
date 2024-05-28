# Smart Fatigue Analysis Of Workers Using Gait Sequence

The objective of this project is to create a reliable, non-intrusive system that uses gait cycle analysis and machine learning to detect fatigue in labor workers. By analyzing patterns in a worker's gait, we aim to identify early signs of fatigue, allowing for proactive intervention and enhanced safety measures.

## Features

- **Gait Cycle Analysis**: Uses MediaPipe for pose estimation to detect knee angle progression during the gait cycle.
- **Fatigue Detection**: Utilizes trajectory data for training the fatigue detection model.
- **Worker Recognition**: Implements face recognition to identify workers using trained face data models.
- **Real-Time Video Feed**: Streams video feed and displays fatigue analysis results and worker identification in real-time.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/mohammad-murasif/Smart-Fatigue-Analysis-Of-Workers--Using-Gait-Sequence.git
    cd Smart-Fatigue-Analysis-Of-Workers--Using-Gait-Sequence
    ```

2. Create a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    ```

3. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Run the Django server:
    ```bash
    python manage.py runserver
    ```

## Usage

1. Open your web browser and go to `http://127.0.0.1:8000`.
2. The main interface will display the video feed.
3. Fatigue analysis results and worker identification will be updated in real-time.

## Project Structure

- `myApp/`: Contains the Django application with necessary views, models, and templates.
- `static/`: Contains static files like CSS and JavaScript.
- `templates/`: Contains HTML templates.
- `requirements.txt`: Lists all Python dependencies.

## Technologies Used

- **Django**: For the web framework.
- **OpenCV**: For video processing.
- **MediaPipe**: For pose estimation.
- **Scikit-Learn**: For machine learning models.
- **TensorFlow**: For deep learning models.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.

## License

MIT License

```plaintext
MIT License

Copyright (c) 2024 Mohammad Murasif

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software with explicit written permission from the copyright holder, 
including without limitation the rights to use, copy, modify, merge, publish, 
distribute, sublicense, and/or sell copies of the Software, and to permit persons 
to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
