import express from 'express';
import cors from 'cors';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import bodyParser from 'body-parser';

// Set up __dirname equivalent for ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 3001;

// Middleware
app.use(cors());
app.use(bodyParser.json());
app.use(express.static(path.join(__dirname, 'public')));

// Path to trading config file
const CONFIG_FILE_PATH = path.join(__dirname, 'public', 'trading_data', 'trading_config.json');

// Endpoint to update trading configuration
app.post('/api/update-trading-config', (req, res) => {
  try {
    // Get the requested interval from the request body
    const { update_interval } = req.body;
    
    if (!update_interval) {
      return res.status(400).json({ error: 'Missing update_interval parameter' });
    }
    
    // Read the current config
    let configData = {};
    if (fs.existsSync(CONFIG_FILE_PATH)) {
      const configContent = fs.readFileSync(CONFIG_FILE_PATH, 'utf8');
      configData = JSON.parse(configContent);
    } else {
      // Default config if file doesn't exist
      configData = {
        update_interval: '5m',
        auto_refresh: true,
        last_modified: new Date().toISOString()
      };
    }
    
    // Update with new interval
    configData.update_interval = update_interval;
    configData.last_modified = new Date().toISOString();
    
    // Make sure directory exists
    const configDir = path.dirname(CONFIG_FILE_PATH);
    if (!fs.existsSync(configDir)) {
      fs.mkdirSync(configDir, { recursive: true });
    }
    
    // Write back to file
    fs.writeFileSync(CONFIG_FILE_PATH, JSON.stringify(configData, null, 2));
    
    console.log(`Trading configuration updated: interval set to ${update_interval}`);
    res.json({ success: true, message: `Updated interval to ${update_interval}` });
    
  } catch (error) {
    console.error('Error updating trading config:', error);
    res.status(500).json({ error: 'Failed to update trading configuration' });
  }
});

// Start server
app.listen(PORT, () => {
  console.log(`Trading config server running at http://localhost:${PORT}`);
});
