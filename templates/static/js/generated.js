function loadImage(maxSize = 1024) {
    // Resize given image file then load to #loadedImage
    let item = document.querySelector('#uploader').files[0]   // Get the given image
    let reader = new FileReader()   // Define a FileReader
    
    reader.readAsDataURL(item)  // Convert to base64 encoded Data URI
    
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


function loadResult(maxSize = 1024) {
    // Resize generated image according to 'maxSize' and load it
    let imagePath = document.querySelector('#result').src   // Get the given image
    let image = new Image()
    image.src = imagePath
    
    image.onload = function (el) {
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

        document.querySelector('#result').src = srcEncoded
    }
}
