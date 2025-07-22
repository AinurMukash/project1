# Указываем базовый образ с Node.js
FROM node:18-alpine

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем package.json и package-lock.json
COPY package*.json ./

# Устанавливаем зависимости
RUN npm install

# Копируем остальной проект
COPY . .

# Собираем приложение
RUN npm run build

# Устанавливаем сервер для предварительного просмотра
RUN npm install -g serve

# Используем порт 8080
EXPOSE 8080

# Команда запуска
CMD ["serve", "-s", "dist", "-l", "8080"]
