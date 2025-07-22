# 1. Этап сборки
FROM node:18-alpine AS build

# 2. Установка рабочей директории
WORKDIR /app

# 3. Копирование зависимостей
COPY package*.json ./

# 4. Установка зависимостей
RUN npm install

# 5. Копирование остальных файлов
COPY . .

# 6. Сборка проекта
RUN npm run build

# 7. Этап запуска
FROM nginx:alpine

# 8. Копируем собранные файлы
COPY --from=build /app/dist /usr/share/nginx/html

# 9. Экспонируем порт
EXPOSE 80

# 10. Запускаем nginx
CMD ["nginx", "-g", "daemon off;"]

