const imageUpload = document.getElementById('imageUpload')

Promise.all([
    // Load model mặc định
    faceapi.nets.ssdMobilenetv1.loadFromUri('./models'),
    faceapi.nets.faceRecognitionNet.loadFromUri('./models'),
    faceapi.nets.faceLandmark68Net.loadFromUri('./models')
]).then(start)

async function start() {
    const LabeledFaceDescriptors = await loadLabeledImages()
    const faceMatcher = new faceapi.FaceMatcher(LabeledFaceDescriptors,0.5) //tạo bản nhận diện khuôn mặt thành tên
    
    document.getElementById('note').innerHTML = 'Đã xử lý xong dữ liệu'

    const imgLoad = document.getElementById('imgload')
    imageUpload.addEventListener('change', async() => { 
        //Xóa ảnh cũ, đưa hình mới lên và điều chỉnh thông số canvas
        imgLoad.innerHTML = ''
        const image = await faceapi.bufferToImage(imageUpload.files[0])
        imgLoad.append(image)
        const canvas = faceapi.createCanvasFromMedia(image)
        imgLoad.append(canvas)
        const displaySize = { width: image.width, height: image.height}
        faceapi.matchDimensions(canvas, displaySize)

        //Cho trình duyệt xử lý, bắt đầu quét hình ảnh
        const detections = await faceapi.detectAllFaces(image).withFaceLandmarks().withFaceDescriptors()
        const resizedDetections = faceapi.resizeResults(detections, displaySize)
        const results = resizedDetections.map(d => faceMatcher.findBestMatch(d.descriptor))
        
        //Xuất Box thông tin quét được  
        results.forEach((result, i) => {
            const box = resizedDetections[i].detection.box
            const drawBox = new faceapi.draw.DrawBox(box, {label: result.toString()})
            drawBox.draw(canvas)
        });
    })
}

function loadLabeledImages() {
    const labels = [['TranThanh', 4], ['LanNgoc', 4], ['JunPham', 3]]  //Tên thư mục hình nhận diện và số lượng hình
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
