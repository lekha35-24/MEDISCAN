from pylibdmtx.pylibdmtx import decode


def decode_datamatrix(image):
    results = decode(image)

    payloads = []
    for r in results:
        try:
            payloads.append(r.data.decode("utf-8"))
        except:
            pass

    return payloads