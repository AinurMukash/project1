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

# 6. Делаем bin-файлы исполняемыми (на всякий случай)
RUN chmod -R +x node_modules/.bin

# 7. Сборка проекта
RUN npm run build

# 8. Этап запуска
FROM nginx:alpine

# 9. Копируем собранные файлы во второй контейнер
COPY --from=build /app/dist /usr/share/nginx/html

# 10. Экспонируем порт
EXPOSE 80

# 11. Запускаем nginx
CMD ["nginx", "-g", "daemon off;"]
