/**
 * Detects if an image likely contains skin and extracts visual signatures.
 * This simulates a CNN preprocessing and feature extraction layer.
 */
export const validateIsSkinImage = (imageSrc) => {
    return new Promise((resolve) => {
        const img = new Image();
        img.src = imageSrc;
        img.onload = () => {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            const scale = 50;
            canvas.width = scale;
            canvas.height = scale;
            ctx.drawImage(img, 0, 0, scale, scale);
            const imageData = ctx.getImageData(0, 0, scale, scale).data;

            let skinPixels = 0;
            let redCount = 0;
            let darkCount = 0;
            let totalLuminance = 0;
            let pixelValues = [];

            for (let i = 0; i < imageData.length; i += 4) {
                const r = imageData[i];
                const g = imageData[i + 1];
                const b = imageData[i + 2];
                const lum = (r + g + b) / 3;
                totalLuminance += lum;
                pixelValues.push(lum);

                // RGB Skin Detection
                const isSkin = (r > 95 && g > 40 && b > 20 &&
                    Math.max(r, g, b) - Math.min(r, g, b) > 15 &&
                    Math.abs(r - g) > 15 && r > g && r > b);

                if (isSkin) {
                    skinPixels++;
                    // Acne/Eczema: Redness detection
                    if (r > g + 40 && r > b + 40) redCount++;
                    // Melanoma: Melanin density (Darkness)
                    if (lum < 75) darkCount++;
                }
            }

            const avgLum = totalLuminance / (scale * scale);
            const stdDev = Math.sqrt(pixelValues.reduce((s, v) => s + Math.pow(v - avgLum, 2), 0) / pixelValues.length);

            resolve({
                isValid: (skinPixels / (scale * scale)) > 0.15,
                signatures: {
                    redness: redCount / (skinPixels || 1),
                    darkness: darkCount / (skinPixels || 1),
                    texture: stdDev // Variance correlates with scaly conditions
                }
            });
        };
    });
};
