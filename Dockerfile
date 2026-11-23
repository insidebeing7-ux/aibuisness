FROM node:20-slim
RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY package*.json requirements.txt ./
RUN npm install --omit=dev && pip3 install --break-system-packages -r requirements.txt
COPY . .
CMD ["node", "server.js"]
