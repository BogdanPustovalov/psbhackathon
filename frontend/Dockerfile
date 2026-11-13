# Stage 1: build the static site.
FROM node:20-alpine AS build

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

# The browser calls the API on the same origin; nginx proxies /api to the backend.
ARG VITE_API_URL=/api/v1
# 1 = demo answers from the built-in mock, without the backend.
ARG VITE_USE_MOCK=0
ENV VITE_API_URL=$VITE_API_URL VITE_USE_MOCK=$VITE_USE_MOCK

RUN npm run build

# Stage 2: serve it with nginx.
FROM nginx:1.27-alpine

# Backend address inside the compose network; substituted into the nginx config at start.
ENV BACKEND_URL=http://backend:8000

COPY nginx.conf.template /etc/nginx/templates/default.conf.template
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80
