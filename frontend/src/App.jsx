import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [text, setText] = useState('');
  const [prediction, setPrediction] = useState(null);

  const handleSubmit = async () => {
    try {
      const res = await axios.post('https://your-backend.up.railway.app/predict', { text });
      setPrediction(res.data.predicted_class);
    } catch (err) {
      console.error(err);
      setPrediction('Ошибка при запросе');
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 p-8">
      <div className="max-w-xl mx-auto bg-white shadow-lg p-6 rounded-xl">
        <h1 className="text-2xl font-bold mb-4">Определение квалификационного уровня в соответствии с Отраслевой рамкой квалификаций (ОРК)</h1>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows="10"
          className="w-full p-10 border rounded mb-4 text-lg"
          placeholder="Введите результат обучения..."
        />
        <button
          onClick={handleSubmit}
          className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
        >
          Предсказать
        </button>
        {prediction && (
          <div className="mt-4 text-lg font-semibold">
            Уровень квалификации: <span className="text-blue-700">{prediction}</span>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
