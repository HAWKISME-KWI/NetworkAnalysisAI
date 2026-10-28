// Địa chỉ backend (Flask/FastAPI...). Có thể đổi bằng biến môi trường: REACT_APP_API_URL=http://host:port/api
export const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';
