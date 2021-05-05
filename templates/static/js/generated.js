function loadImage(maxSize=1024) {
    // Load image file from the disk
    // Fill img with id 'loadedImage'
    let item = document.querySelector('#uploader').files[0]   // Get the given image
    let reader = new FileReader()   // Define a FileReader
    
    reader.readAsDataURL(item)  // Convert to base64 encoded Data URI
    reader.size = item.size     // Get image's size

    reader.onload = function (event) {
        let img = new Image()
        img.src = event.target.result

        img.onload = function (el) {
            let elem = document.createElement('canvas')

            if (el.target.width > maxSize || el.target.height > maxSize) {
                let scaleFactor = null
                if (el.target.width >= el.target.height) {
                    scaleFactor = maxSize / el.target.width
                    elem.width = maxSize
                    elem.height = el.target.height * scaleFactor
                } else {
                    scaleFactor = maxSize / el.target.height
                    elem.width = el.target.width * scaleFactor
                    elem.height = maxSize
                    console.log("image shapes: ", el.target.width, el.target.height)
                    console.log("canvas shapes: ", elem.width, elem.height)
                }
            } else {
                elem.width = el.target.width
                elem.height = el.target.height
            }

            let ctx = elem.getContext('2d')
            ctx.drawImage(el.target, 0, 0, elem.width, elem.height)

            let srcEncoded = ctx.canvas.toDataURL(el.target, 'image/jpeg', 0)

            document.querySelector('#loadedImage').src = srcEncoded
        }
    }
}
