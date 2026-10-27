import avatarUrl from "../../models/avatar.glb?url";
import maleUrl from "../../models/visualmind-male.glb?url";
import femaleUrl from "../../models/visualmind-female.glb?url";
import waterUrl from "../../models/вода.glb?url";
import deafUrl from "../../models/глухой.glb?url";
import diplomaUrl from "../../models/диплом подправить левая рука.glb?url";
import showerUrl from "../../models/душ.glb?url";
import internetUrl from "../../models/интернет подправить пальцы.glb?url";
import ditchUrl from "../../models/канава.glb?url";
import educationUrl from "../../models/образование подправить голову.glb?url";
import windowUrl from "../../models/окно.glb?url";
import writeUrl from "../../models/писать.glb?url";
import developmentUrl from "../../models/развитие.glb?url";
import craneUrl from "../../models/строительный кран подправить.glb?url";
import toiletUrl from "../../models/туалет.glb?url";

export type AvatarOption = {
  id: string;
  label: string;
  url: string;
};

export const AVATAR_OPTIONS: AvatarOption[] = [
  { id: "visualmind-male", label: "Visual Mind male", url: maleUrl },
  { id: "visualmind-female", label: "Visual Mind female", url: femaleUrl },
  { id: "avatar", label: "avatar", url: avatarUrl },
  { id: "water", label: "вода", url: waterUrl },
  { id: "deaf", label: "глухой", url: deafUrl },
  { id: "diploma-left-hand", label: "диплом", url: diplomaUrl },
  { id: "shower", label: "душ", url: showerUrl },
  { id: "internet-fingers", label: "интернет", url: internetUrl },
  { id: "ditch", label: "канава", url: ditchUrl },
  { id: "education-head", label: "образование", url: educationUrl },
  { id: "window", label: "окно", url: windowUrl },
  { id: "write", label: "писать", url: writeUrl },
  { id: "development", label: "развитие", url: developmentUrl },
  { id: "crane", label: "строительный кран", url: craneUrl },
  { id: "toilet", label: "туалет", url: toiletUrl },
];
