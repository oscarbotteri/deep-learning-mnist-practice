import { useEffect, useCallback, useRef, useState, MouseEvent, TouchEvent } from "react";
import * as tf from '@tensorflow/tfjs';

interface Coordinate {
  x: number;
  y: number;
}

const WIDTH = 200
const HEIGHT = 250;

function App() {
  const canvas = useRef<HTMLCanvasElement>(null)
  const [drawing, setDrawing] = useState<boolean>(false);
  const [position, setPosition] = useState<Coordinate>({ x: 0, y: 0 });
  const [model, setModel] = useState<tf.LayersModel | undefined>(undefined);
  const [prediction, setPrediction] = useState<number | undefined>(undefined);
  const [image, setImage] = useState<string | undefined>(undefined);

  useEffect(() => {
    tf
      .loadLayersModel(`${import.meta.env.BASE_URL}model.json`)
      .then(setModel)
      .catch(() => alert('Oops! Error loading AI model'));
  }, []);

  const onDown = useCallback((event: MouseEvent<HTMLCanvasElement> | TouchEvent<HTMLCanvasElement>) => {
    const coordinates = getCoordinates(event);

    setDrawing(true);
    setPosition(coordinates);
  }, []);

  const onUp = useCallback((event: MouseEvent<HTMLCanvasElement> | TouchEvent<HTMLCanvasElement>) => {
    const coordinates = getCoordinates(event);

    setDrawing(false);
    setPosition(coordinates);
  }, []);

  const onMove = useCallback(
    (event: MouseEvent<HTMLCanvasElement> | TouchEvent<HTMLCanvasElement>) => {
      if (!drawing) {
        return;
      }

      const newPosition = getCoordinates(event);

      drawLine(position, newPosition);
      setPosition(newPosition);
    },
    [drawing, position]
  );

  const isMouseEvent = (event: TouchEvent | MouseEvent): event is MouseEvent => {
    return event && 'screenX' in event;
  }

  const getCoordinates = (event: MouseEvent<HTMLCanvasElement> | TouchEvent<HTMLCanvasElement>): Coordinate => {
    if (!canvas.current) {
      return { x: 0, y: 0 };
    }

    const x = isMouseEvent(event) ? event.pageX : event.touches[0].pageX;
    const y = isMouseEvent(event) ? event.pageY : event.touches[0].pageY;

    return {
      x: x - canvas.current.offsetLeft,
      y: y - canvas.current.offsetTop
    }
  }

  const drawLine = (originalPosition: Coordinate, newPosition: Coordinate) => {
    if (!canvas.current) {
      return;
    }

    const context = canvas.current.getContext('2d')!;

    context.strokeStyle = 'red';
    context.lineJoin = 'round';
    context.lineWidth = 8;

    context.beginPath();
    context.moveTo(originalPosition.x, originalPosition.y);
    context.lineTo(newPosition.x, newPosition.y);
    context.closePath();

    context.stroke();
  }

  const clear = () => {
    setDrawing(false);
    setPrediction(undefined);

    if (!canvas.current) {
      return;
    }

    const context = canvas.current.getContext('2d')!;

    context.clearRect(0, 0, WIDTH, HEIGHT);
  }

  const predict = () => {
    if (!model || !canvas.current) {
      return;
    }

    const newCanvas = document.createElement('canvas')!;
    const newContext = newCanvas.getContext("2d")!;

    newCanvas.width = 28;
    newCanvas.height = 28;
    newContext.drawImage(canvas.current, 0, 0, 28, 28);

    let tmp = [];
    const data = [];
    const image = newContext.getImageData(0 ,0 , 28, 28);

    setImage(newCanvas.toDataURL('image/png'));

    for (let p= 0; p < image.data.length; p+= 4) {
      const value = image.data[p + 3] / 255;
      tmp.push([value]);

      if (tmp.length === 28) {
        data.push(tmp);
        tmp = [];
      }
    }

    const tensor4 = tf.tensor4d([ data ]);
    const prediction = model.predict(tensor4).dataSync();
    const mostSimilar = prediction.indexOf(Math.max.apply(null, prediction));

    setPrediction(mostSimilar);
  }

  return (
    <main className="p-4">
      <div className="d-flex flex-column justify-content-center align-items-center">
        <div className="w50 text-light text-center">
          <h1>Number predictor</h1>
          <p className="fs-4">
            Small React application using <span className="badge bg-primary">tensorflowjs</span>
            to load a trained neural<br />
            network model using <span className="badge bg-primary">mmist</span>
            dataset to predict handwritten digits.
          </p>
        </div>
      </div>

      <div className="d-flex justify-content-center align-items-center flex-column mt-4">
        <canvas
          className="rounded bg-white"
          ref={canvas}
          onMouseDown={onDown}
          onTouchStart={onDown}
          onMouseUp={onUp}
          onTouchEnd={onUp}
          onMouseLeave={onUp}
          onMouseMove={onMove}
          onTouchMove={onMove}
          width={WIDTH}
          height={HEIGHT}
        />

        {!!prediction && (
          <div className="d-flex justify-content-center align-items-center mt-4">
            <img src={image} className="bg-white rounded" />
            <span className="text-white mx-1">=</span>
            <div className="text-primary bg-white rounded py-1 px-2">{prediction}</div>
          </div>
        )}

        <div className="d-flex justify-content-center align-items-center mt-4">
          <button role="button" className="btn btn-danger me-2" disabled={drawing} onClick={clear}>
            Clear
          </button>

          <button role="button" className="btn btn-primary ms-2" disabled={drawing || !model} onClick={predict}>
            Predict
          </button>
        </div>
      </div>
    </main>
  )
}

export default App
