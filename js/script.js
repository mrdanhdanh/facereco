const imageUpload = document.getElementById('imageUpload')

Promise.all([
    // Load model
    faceapi.nets.ssdMobilenetv1.loadFromUri('./models'),
    faceapi.nets.faceRecognitionNet.loadFromUri('./models'),
    faceapi.nets.faceLandmark68Net.loadFromUri('./models')
]).then(start)

async function start() {
    const LabeledFaceDescriptors = await loadLabeledImages()
    const faceMatcher = new faceapi.FaceMatcher(LabeledFaceDescriptors,0.5)
    
    document.getElementById('note').innerHTML = 'Đã tải xong dữ liệu'
    const imgLoad = document.getElementById('imgload')
    imageUpload.addEventListener('change', async() => {
        imgLoad.innerHTML = ''
        const image = await faceapi.bufferToImage(imageUpload.files[0])
        imgLoad.append(image)
        const canvas = faceapi.createCanvasFromMedia(image)
        imgLoad.append(canvas)
        const displaySize = { width: image.width, height: image.height}
        faceapi.matchDimensions(canvas, displaySize)

        const detections = await faceapi.detectAllFaces(image)
        .withFaceLandmarks().withFaceDescriptors()
        const resizedDetections = faceapi.resizeResults(detections, displaySize)
        const results = resizedDetections.map(d => faceMatcher.findBestMatch(d.descriptor))
        
        //faceapi.draw.drawDetections(canvas, resizedDetections)

        results.forEach((result, i) => {
            const box = resizedDetections[i].detection.box
            const drawBox = new faceapi.draw.DrawBox(box, {label: result.toString()})
            drawBox.draw(canvas)
        });
    })
}

    function loadLabeledImages() {
    const labels = [['TranThanh', 4], ['LanNgoc', 4], ['JunPham', 3]] 
    return Promise.all(
        labels.map(async per => {
            const descriptions = []
            for (let i=1; i<=per[1]; i++) {
                const img  = await faceapi.fetchImage(`./images/${per[0]}/${i}.jpg`)
                const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor()
                descriptions.push(detections.descriptor)
            }
            return new faceapi.LabeledFaceDescriptors(per[0], descriptions)
        })
    )
}
//     return Promise.all(
//         labels.map(async label => {
//             const descriptions = []
//             for (let i=1; i<=1; i++) {
//                 const img  = await faceapi.fetchImage(`./images/${label}/${i}.jpg`)
//                 const detections = await faceapi.detectSingleFace(img)
//                 .withFaceLandmarks().withFaceDescriptors()
//                 descriptions.push(detections.descriptor)
//             }
//             return new faceapi.LabeledFaceDescriptors(label, descriptions)
//         })
//     )
// }