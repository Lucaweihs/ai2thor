$(() => {
  let getParams = parseGet();
  function paramStrToAssocArray(prmstr) {
    let params = {};
    let prmarr = prmstr.split('&');
    for (let i = 0; i < prmarr.length; i++) {
      let tmparr = prmarr[i].split('=');
      params[tmparr[0]] = tmparr[1];
    }
    return params;
  }

  function parseGet() {
    let paramStr = window.location.search.substr(1);
    return paramStr !== null && paramStr !== ''
      ? paramStrToAssocArray(paramStr)
      : {};
  }

  function redirect(role) {
    let url_str = location.protocol + '//' + location.host + location.pathname;
    let url = window.location.href.split('/');
    url[url.length-1] = "player.html";
    let redirect = url.join("/") ;
    console.log(url);
    // window.location.pathname = "/player.html"
    window.location = `${redirect}?role=${role}&random=true&game_set=${getParams['game_set']}`;

  }



  $("#seeker-btn").click((e) => {
    redirect('seeker');
  })

})