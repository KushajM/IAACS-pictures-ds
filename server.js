import express from 'express';
import path from 'path';
import fs from 'fs';
import fetch from 'node-fetch';

const app = express();
const port = 3000;

app.use(express.static('public'));

app.get('/save-image', async (req, res) => {
    const { url, name } = req.query;
    if (!url || !name) {
        return res.json({ success: false, message: 'Failed to save image' });
    }

    try {
        const response = await fetch(url);
        const buffer = await response.buffer();
        const filePath = path.join(process.cwd(), 'images', name);

        fs.writeFile(filePath, buffer, () => {
            res.json({ success: true });
        });
    } catch (error) {
        console.error(error);
        res.json({ success: false, message: 'Failed to save image' });
    }
});

app.listen(port, () => {
    console.log(`Sv http://localhost:${port}`);
});
