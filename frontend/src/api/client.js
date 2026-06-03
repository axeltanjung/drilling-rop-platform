import axios from 'axios'

const baseURL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

const client = axios.create({ baseURL, timeout: 30000 })

export const api = {
  health: () => client.get('/health').then((r) => r.data),
  dashboardSummary: () => client.get('/dashboard/summary').then((r) => r.data),
  well: (id) => client.get(`/well/${id}`).then((r) => r.data),
  riskOverview: () => client.get('/risk/overview').then((r) => r.data),
  predictRop: (payload) => client.post('/predict/rop', payload).then((r) => r.data),
  predictRisk: (payload) => client.post('/predict/risk', payload).then((r) => r.data),
  predictEfficiency: (payload) => client.post('/predict/efficiency', payload).then((r) => r.data),
  optimize: (payload) => client.post('/optimize/drilling', payload).then((r) => r.data),
  explain: (payload) => client.post('/explain-prediction', payload).then((r) => r.data),
  exportReportUrl: `${baseURL}/export/optimization-report`,
  exportCsvUrl: `${baseURL}/export/predictions.csv`,
  downloadReport: (payload) =>
    client.post('/export/optimization-report', payload, { responseType: 'blob' }).then((r) => r.data),
}

export default client
