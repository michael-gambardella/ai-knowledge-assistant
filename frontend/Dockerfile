# Stage 1: Build the React app
# Use a full Node image to compile TypeScript and bundle with Vite
FROM node:20-slim AS build

WORKDIR /app

COPY package*.json ./
RUN npm ci

COPY . .
RUN npm run build

# Stage 2: Serve with nginx
# The final image is just nginx + the compiled static files (~20MB total)
# vs keeping the Node image (~600MB). Multi-stage builds are the standard
# pattern for frontend Docker images.
FROM nginx:alpine

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
