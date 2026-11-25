"""Generate a small synthetic manual + gold questions so the pipeline can be smoke-tested.
Replace with real manuals (data/manuals/*.pdf) and a hand-labelled data/eval/questions.jsonl."""
import json
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, PageBreak, Spacer
from reportlab.lib import colors

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / "data/manuals/sample_router_x200.pdf"
out.parent.mkdir(parents=True, exist_ok=True)
ss = getSampleStyleSheet()
H, B = ss["Heading1"], ss["BodyText"]

def tbl(rows):
    t = Table(rows, hAlign="LEFT")
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black),
                           ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)]))
    return t

story = [
    Paragraph("NetRouter X200 User Manual", H),
    Paragraph("The NetRouter X200 is a dual-band wireless router with four gigabit LAN ports. "
              "Place the router in a ventilated area away from heat sources. Use only the supplied "
              "12V power adapter. WARNING: Never open the casing while the unit is connected to power.", B),
    PageBreak(),
    Paragraph("Chapter 2: Status LED Indicators", H),
    Paragraph("The status LED on the front panel shows the operating state of the router.", B),
    Spacer(1, 8),
    tbl([["LED pattern", "Meaning", "Action"],
         ["Solid green", "Normal operation", "None"],
         ["Blinking green", "Booting up", "Wait 90 seconds"],
         ["Red blinking 3 times", "WAN link failure", "Check the WAN cable and restart the modem"],
         ["Red blinking 5 times", "Overheating", "Move the router to a cooler place and clear the vents"],
         ["Solid amber", "Firmware update in progress", "Do not power off"]]),
    PageBreak(),
    Paragraph("Chapter 3: Error Codes", H),
    Paragraph("Error codes appear in the web admin page under Diagnostics.", B),
    Spacer(1, 8),
    tbl([["Code", "Description", "Resolution"],
         ["E-203", "DHCP lease could not be obtained", "Reboot the modem, then renew the lease from Diagnostics"],
         ["E-417", "Firmware checksum mismatch", "Re-download the firmware image and flash it again"],
         ["E-509", "5 GHz radio fault", "Perform a factory reset; contact support if it persists"]]),
    PageBreak(),
    Paragraph("Chapter 4: Factory Reset", H),
    Paragraph("To restore factory settings, hold the recessed RESET button for ten seconds using a "
              "paperclip until the status LED turns solid amber. Release the button and wait for the "
              "router to reboot. All custom settings, including the Wi-Fi password, will be erased.", B),
    PageBreak(),
    Paragraph("Chapter 5: Connecting to the Internet", H),
    Paragraph("Connect the WAN port to your modem with an Ethernet cable. Power on the router and "
              "open the admin page at 192.168.8.1. The setup wizard will detect the connection type "
              "automatically. If your provider uses PPPoE, enter the username and password supplied.", B),
]
SimpleDocTemplate(str(out), pagesize=A4).build(story)

gold = lambda p: [{"manual": "sample_router_x200", "page": p}]
qs = [
    {"question": "The status LED is blinking red three times, what does it mean?", "type": "table", "gold": gold(2)},
    {"question": "My router shows red LED flashing 5 times and feels hot", "type": "table", "gold": gold(2)},
    {"question": "What does error E-417 mean?", "type": "table", "gold": gold(3)},
    {"question": "I can't get an IP address from my ISP, DHCP lease failed", "type": "table", "gold": gold(3)},
    {"question": "How do I reset the router to factory defaults?", "type": "text", "gold": gold(4)},
    {"question": "How do I log in to the admin page?", "type": "text", "gold": gold(5)},
    {"question": "What power adapter should I use?", "type": "text", "gold": gold(1)},
    {"question": "How do I replace the printer toner cartridge?", "type": "unanswerable", "gold": []},
]
qf = ROOT / "data/eval/questions.jsonl"
qf.parent.mkdir(parents=True, exist_ok=True)
qf.write_text("\n".join(json.dumps(q) for q in qs) + "\n")
print("wrote", out, "and", qf)
