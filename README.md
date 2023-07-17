# Deep Learning mnist Practice

My first NN to using `tensorflow` to classify images using `mnist` dataset.
Available UI to test the model

## How to run

Install dependencies from `requirements.txt`
```shell
$ pip install -r requirements.txt
```

Run the script:
```shell
$ python main.py
```

# Convert model
In order to load the generated model in a React app, we have to convert the model using `tensorflowjs`
```shell
$ tensorflowjs_converter --input_format keras \
    ./model.h5 \
    ./dist
```

# Run the web app
Copy the content from `dist` to `ui/public` install dependencies and run the web app:
```
$ cd ui
$ npm i
$ npm run dev
```