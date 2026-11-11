import axios from 'axios'

const BASE = import.meta.env.VITE_API_URL || '/api'

const client = axios.create({ baseURL: BASE, timeout: 30000 })

export const getHealth = () => client.get('/health').then(r => r.data)

export const predict = (customer, threshold = 0.4) =>
  client.post('/predict', { customer, threshold }).then(r => r.data.prediction)

export const predictBatch = (customers, threshold = 0.4) =>
  client.post('/predict/batch', { customers, threshold }).then(r => r.data)
