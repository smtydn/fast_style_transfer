function loadContentImage(maxSize = 1024) {
    // Resize given image file then load to #loadedImage
    let item = document.querySelector('#uploader').files[0]   // Get the given image
    let reader = new FileReader()   // Define a FileReader
    
    reader.readAsDataURL(item)  // Convert to base64 encoded Data URI
    
    reader.onload = function (event) {
        let img = new Image()
        img.src = event.target.result
        img.onload = (event) => { document.querySelector('#loadedImage').src = _resizeImage(event, maxSize) }
    }
}


function loadGeneratedImage(maxSize = 1024) {
    // Resize generated image according to 'maxSize' and load it to '#result'
    let image = new Image()
    image.src = document.querySelector('#result').src
    image.onload = (event) => { document.querySelector('#result').src = _resizeImage(event, maxSize) }
}


function _resizeImage(event, size) {
    // If wdith or height of the image is bigger than 'size',
    // scale image to make its dimensions <= size
    let elem = document.createElement('canvas')

    if (event.target.width > size || event.target.height > size) {
        let scaleFactor = null
        if (event.target.width >= event.target.height) {
            scaleFactor = size / event.target.width
            elem.width = size
            elem.height = event.target.height * scaleFactor
        } else {
            scaleFactor = size / event.target.height
            elem.width = event.target.width * scaleFactor
            elem.height = size
        }
    } else {
        elem.width = event.target.width
        elem.height = event.target.height
    }

    let ctx = elem.getContext('2d')
    ctx.drawImage(event.target, 0, 0, elem.width, elem.height)

    return ctx.canvas.toDataURL(event.target, 'image/jpeg', 0)
}