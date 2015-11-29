$(document).ready(function() {

  var notReady = true;
  var interaction = null;

  $("#a").click(function() {
    send(0);
  })
  $("#b").click(function() {
    send(1);
  })
  $("#c").click(function() {
    send(2);
  })
  $("#noidea").click(function() {
    fetch();
  })

  var send = function(idx) {
    if (notReady) {
      console.log("Not ready")
      return;
    }
    notReady = true;
    $.post("/api/add_interaction", {
      'a': interaction[0],
      'b': interaction[1],
      'c': interaction[2],
      'oou': idx
    },
    function() {
      notReady = true;
      fetch();
    });
  }
  var fetch = function() {
    $.getJSON("/api/fetch_interaction", function(data) {
      data = data['interaction']
      keys = Object.keys(data).sort()
      interaction = keys;
      for (var i in keys) {
        key = keys[i];
        if (i == 0) {
          $("#a").html(data[keys[i]])
        }
        if (i == 1) {
          $("#b").html(data[keys[i]])
        }
        if (i == 2) {
          $("#c").html(data[keys[i]])
        }
      }
      notReady = false;
    });
  }

  fetch();

});
