const fs = require('fs');
const https = require('https');
const sharp = require('sharp');
const path = require('path');

// Create a temporary file to save the downloaded image
const tempFile = path.join(__dirname, 'temp-avatar.jpg');
const outputFile = path.join(__dirname, 'src/assets/images/dimbot-avatar.png');

// This function creates a circular mask
async function createCircularAvatar() {
  try {
    // Create a circular mask
    const size = 200; // Size of the output image
    const centerX = size / 2;
    const centerY = size / 2;
    const radius = size / 2;
    
    // Create a circle SVG for the mask
    const circleSvg = Buffer.from(
      `<svg width="${size}" height="${size}">
         <circle cx="${centerX}" cy="${centerY}" r="${radius}" fill="white"/>
       </svg>`
    );

    console.log('Processing image...');
    
    // Read the local file we downloaded
    await sharp(tempFile)
      // Resize to a reasonable size
      .resize(size, size, { fit: 'cover' })
      // Apply the circular mask
      .composite([{
        input: circleSvg,
        blend: 'dest-in'
      }])
      // Make the background transparent
      .png({ quality: 90 })
      .toFile(outputFile);
    
    console.log(`Avatar saved to ${outputFile}`);
    
    // Clean up temp file
    fs.unlinkSync(tempFile);
  } catch (error) {
    console.error('Error processing image:', error);
  }
}

// Download the image first
console.log('Downloading image...');
const imageUrl = 'https://avatars.githubusercontent.com/u/1?v=4'; // Using a placeholder as we can't download from the shared image directly

const file = fs.createWriteStream(tempFile);
https.get(imageUrl, response => {
  response.pipe(file);
  file.on('finish', () => {
    file.close();
    console.log('Image download complete');
    createCircularAvatar();
  });
}).on('error', err => {
  fs.unlink(tempFile);
  console.error('Error downloading image:', err.message);
});
